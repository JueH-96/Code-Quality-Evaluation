import sys
import threading
from collections import defaultdict, deque

def main():
    sys.setrecursionlimit(1 << 25)
    N, M, K = map(int, sys.stdin.readline().split())
    edges = []
    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        edges.append((w, u, v))
    
    # Kruskal's algorithm to build MST
    class DSU:
        def __init__(self, size):
            self.parent = list(range(size + 1))
            self.rank = [0] * (size + 1)
        def find(self, x):
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x
        def union(self, x, y):
            x_root = self.find(x)
            y_root = self.find(y)
            if x_root == y_root:
                return False
            if self.rank[x_root] < self.rank[y_root]:
                self.parent[x_root] = y_root
            else:
                self.parent[y_root] = x_root
                if self.rank[x_root] == self.rank[y_root]:
                    self.rank[x_root] += 1
            return True
    
    dsu = DSU(N)
    mst_edges = []
    # Sort edges by weight
    edges.sort()
    for w, u, v in edges:
        if dsu.union(u, v):
            mst_edges.append((u, v, w))
            mst_edges.append((v, u, w))
    
    # Build adjacency list for the MST
    adj = defaultdict(list)
    for u, v, w in mst_edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
    
    # Preprocess for LCA with maximum edge weight
    LOG = 20
    up = [[-1] * LOG for _ in range(N + 1)]
    max_edge = [[0] * LOG for _ in range(N + 1)]
    depth = [0] * (N + 1)
    
    # BFS to set up parent, depth, and initial max_edge
    visited = [False] * (N + 1)
    q = deque()
    root = 1
    q.append((root, -1, 0))  # node, parent, depth
    visited[root] = True
    while q:
        u, parent, d = q.popleft()
        depth[u] = d
        up[u][0] = parent
        for v, w in adj[u]:
            if not visited[v]:
                visited[v] = True
                q.append((v, u, d + 1))
                max_edge[v][0] = w
    
    # Build binary lifting table
    for k in range(1, LOG):
        for v in range(1, N + 1):
            if up[v][k-1] != -1:
                up[v][k] = up[up[v][k-1]][k-1]
                max_edge[v][k] = max(max_edge[v][k-1], max_edge[up[v][k-1]][k-1])
    
    # Function to compute maximum edge between u and v
    def get_max(u, v):
        if u == v:
            return 0
        max_val = 0
        if depth[u] < depth[v]:
            u, v = v, u
        # Bring u up to the depth of v
        for k in range(LOG-1, -1, -1):
            if depth[u] - (1 << k) >= depth[v]:
                max_val = max(max_val, max_edge[u][k])
                u = up[u][k]
        if u == v:
            return max_val
        # Now bring both up until LCA is found
        for k in range(LOG-1, -1, -1):
            if up[u][k] != -1 and up[u][k] != up[v][k]:
                max_val = max(max_val, max_edge[u][k], max_edge[v][k])
                u = up[u][k]
                v = up[v][k]
        # Now u and v are children of LCA
        max_val = max(max_val, max_edge[u][0], max_edge[v][0])
        return max_val
    
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    
    # Build cost matrix
    cost = []
    for a in A:
        row = []
        for b in B:
            row.append(get_max(a, b))
        cost.append(row)
    
    # Hungarian algorithm for assignment problem
    def hungarian(matrix):
        n = len(matrix)
        m = len(matrix[0])
        assert n == m
        u = [0] * (n + 1)
        v = [0] * (n + 1)
        p = [0] * (n + 1)
        way = [0] * (n + 1)
        for i in range(1, n + 1):
            p[0] = 0
            way[0] = 0
            j0 = 0
            minv = [float('inf')] * (n + 1)
            used = [False] * (n + 1)
            while True:
                used[j0] = True
                j0 = p[j0]
                if j0 == 0:
                    break
                i0 = i
                delta = float('inf')
                j1 = 0
                for j in range(1, n + 1):
                    if not used[j]:
                        cur = -matrix[i0-1][j-1] - u[i0] - v[j]
                        if cur < minv[j]:
                            minv[j] = cur
                            way[j] = j0
                        if minv[j] < delta:
                            delta = minv[j]
                            j1 = j
                for j in range(n + 1):
                    if used[j]:
                        u[j] += minv[j]
                    v[j] += minv[j]
                    if used[j]:
                        p[j] = way[j]
                j0 = j1
            while j0 != 0:
                p[j0] = way[j0]
                j1 = p[j0]
                u[i], v[j0] = u[i] - matrix[i-1][j0-1], v[j0] + matrix[i-1][j0-1]
                i, j0 = p[j1], 0
        return -v[0]
    
    res = hungarian(cost)
    print(res)

threading.Thread(target=main).start()