import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N, M, K = map(int, sys.stdin.readline().split())
    edges = []
    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        edges.append((w, u-1, v-1))
    edges.sort()
    
    # DSU to build MST
    parent = list(range(N))
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    def union(u, v):
        u_root = find(u)
        v_root = find(v)
        if u_root == v_root:
            return False
        parent[v_root] = u_root
        return True
    
    mst_edges = set()
    for w, u, v in edges:
        if union(u, v):
            if u < v:
                mst_edges.add((u, v))
            else:
                mst_edges.add((v, u))
    
    # Build adjacency list for MST
    adj = [[] for _ in range(N)]
    for u, v in mst_edges:
        adj[u].append((v, edges[edges.index((u, v))][2]))
        adj[v].append((u, edges[edges.index((u, v))][2]))
    
    # BFS to assign parent and depth
    root = 0
    parent = [-1] * N
    depth = [0] * N
    visited = [False] * N
    q = deque()
    q.append(root)
    visited[root] = True
    parent[root] = -1
    while q:
        u = q.popleft()
        for v, w in adj[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                depth[v] = depth[u] + 1
                q.append(v)
    
    # Build binary lifting tables
    log_max = 20
    up = [[-1] * N for _ in range(log_max)]
    max_edge = [[0] * N for _ in range(log_max)]
    
    for u in range(N):
        if parent[u] == -1:
            up[0][u] = -1
            max_edge[0][u] = 0
        else:
            up[0][u] = parent[u]
            max_edge[0][u] = edges[edges.index((u, parent[u]))][2] if u < parent[u] else edges[edges.index((parent[u], u))][2]
    
    for k in range(1, log_max):
        for u in range(N):
            up_prev = up[k-1][u]
            if up_prev == -1:
                up[k][u] = -1
                max_edge[k][u] = 0
            else:
                up[k][u] = up[k-1][up_prev]
                max_edge[k][u] = max(max_edge[k-1][u], max_edge[k-1][up_prev])
    
    def get_max_edge(u, v):
        max_e = 0
        if depth[u] < depth[v]:
            u, v = v, u
        # Bring u up to the depth of v
        for k in range(log_max-1, -1, -1):
            if depth[u] - (1 << k) >= depth[v]:
                max_e = max(max_e, max_edge[k][u])
                u = up[k][u]
        if u == v:
            return max_e
        # Now find LCA
        for k in range(log_max-1, -1, -1):
            if up[k][u] != -1 and up[k][u] != up[k][v]:
                max_e = max(max_e, max_edge[k][u], max_edge[k][v])
                u = up[k][u]
                v = up[k][v]
        # Finally, compare the edges to LCA
        max_e = max(max_e, max_edge[0][u], max_edge[0][v])
        return max_e
    
    # Read A and B
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    
    # Sort A increasing, B decreasing
    A_sorted = sorted(A)
    B_sorted = sorted(B, reverse=True)
    
    total = 0
    for a, b in zip(A_sorted, B_sorted):
        u = a - 1
        v = b - 1
        total += get_max_edge(u, v)
    
    print(total)

if __name__ == "__main__":
    main()