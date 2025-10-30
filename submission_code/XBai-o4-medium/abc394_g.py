import sys
import threading
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    H, W = map(int, sys.stdin.readline().split())
    F = [list(map(int, sys.stdin.readline().split())) for _ in range(H)]
    
    edges = []
    for i in range(H):
        for j in range(W):
            if j + 1 < W:
                u = i * W + j
                v = i * W + (j+1)
                w = min(F[i][j], F[i][j+1])
                edges.append((-w, u, v))
            if i + 1 < H:
                u = i * W + j
                v = (i+1)* W + j
                w = min(F[i][j], F[i+1][j])
                edges.append((-w, u, v))
    
    edges.sort()
    
    class DSU:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [1]*n
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
    
    dsu = DSU(H * W)
    tree_adj = [[] for _ in range(H * W)]
    for e in edges:
        w = -e[0]
        u = e[1]
        v = e[2]
        if dsu.union(u, v):
            tree_adj[u].append((v, w))
            tree_adj[v].append((u, w))
    
    LOG = 20
    MAXN = H * W
    up = [[-1] * MAXN for _ in range(LOG)]
    min_edge_up = [[0] * MAXN for _ in range(LOG)]
    depth = [0] * MAXN
    
    root = 0
    q = deque()
    q.append(root)
    visited = [False] * (H * W)
    visited[root] = True
    parent_bfs = [-1] * (H * W)
    min_edge_bfs = [0] * (H * W)
    
    while q:
        u = q.popleft()
        for v, w in tree_adj[u]:
            if not visited[v] and v != parent_bfs[u]:
                visited[v] = True
                parent_bfs[v] = u
                depth[v] = depth[u] + 1
                min_edge_bfs[v] = w
                q.append(v)
    
    for v in range(MAXN):
        up[0][v] = parent_bfs[v]
        if up[0][v] != -1:
            min_edge_up[0][v] = min_edge_bfs[v]
        else:
            min_edge_up[0][v] = F[0][0]
    
    for k in range(1, LOG):
        for v in range(MAXN):
            if up[k-1][v] != -1:
                up[k][v] = up[k-1][up[k-1][v]]
                min_edge_up[k][v] = min(min_edge_up[k-1][v], min_edge_up[k-1][up[k-1][v]])
            else:
                up[k][v] = -1
                min_edge_up[k][v] = min_edge_up[k-1][v]
    
    def get_min_edge(u, v):
        min_e = float('inf')
        if u == v:
            return F[u // W][u % W]
        if depth[u] < depth[v]:
            u, v = v, u
        for k in range(LOG-1, -1, -1):
            if depth[u] - (1 << k) >= depth[v]:
                min_e = min(min_e, min_edge_up[k][u])
                u = up[k][u]
        if u == v:
            return min_e
        for k in range(LOG-1, -1, -1):
            if up[k][u] != -1 and up[k][u] != up[k][v]:
                min_e = min(min_e, min_edge_up[k][u], min_edge_up[k][v])
                u = up[k][u]
                v = up[k][v]
        min_e = min(min_e, min_edge_up[0][u], min_edge_up[0][v])
        return min_e
    
    Q = int(sys.stdin.readline())
    for _ in range(Q):
        A, B, Y, C, D, Z = map(int, sys.stdin.readline().split())
        u = (A-1)*W + (B-1)
        v = (C-1)*W + (D-1)
        if u == v:
            x_max = F[A-1][B-1]
        else:
            x_max = get_min_edge(u, v)
        L = min(Y, Z)
        R = max(Y, Z)
        if x_max >= L:
            print(R - L)
        else:
            print(Y + Z - 2 * x_max)

threading.Thread(target=main).start()