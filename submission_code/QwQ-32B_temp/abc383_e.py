import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N, M, K = map(int, sys.stdin.readline().split())
    edges = []
    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        edges.append((w, u, v))
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    
    # Kruskal's algorithm to find MST
    edges.sort()
    parent = list(range(N+1))
    rank = [1]*(N+1)
    
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]  # Path compression
            u = parent[u]
        return u
    
    mst_edges = []
    adj = [[] for _ in range(N+1)]
    for w, u, v in edges:
        a = find(u)
        b = find(v)
        if a != b:
            mst_edges.append((u, v, w))
            if rank[a] < rank[b]:
                parent[a] = b
            else:
                parent[b] = a
                if rank[a] == rank[b]:
                    rank[a] += 1
            # Add to adjacency list both ways
            adj[u].append((v, w))
            adj[v].append((u, w))
    
    # Build binary lifting tables for LCA
    LOG = 20
    up = [[0]*(N+1) for _ in range(LOG)]
    max_edge_table = [[0]*(N+1) for _ in range(LOG)]
    depth = [0]*(N+1)
    
    # Initialize with BFS from root 1
    root_initial = 1
    visited = [False]*(N+1)
    q = deque()
    q.append(root_initial)
    visited[root_initial] = True
    up[0][root_initial] = 0  # parent of root is 0
    depth[root_initial] = 0
    while q:
        u = q.popleft()
        for v, w in adj[u]:
            if not visited[v] and v != up[0][u]:
                up[0][v] = u
                depth[v] = depth[u] + 1
                max_edge_table[0][v] = w
                visited[v] = True
                q.append(v)
    
    # Fill the binary lifting tables
    for k in range(1, LOG):
        for v in range(1, N+1):
            up[k][v] = up[k-1][up[k-1][v]]
            max_edge_table[k][v] = max(max_edge_table[k-1][v], max_edge_table[k-1][up[k-1][v]])
    
    # LCA function
    def lca(u, v):
        if depth[u] < depth[v]:
            u, v = v, u
        # Bring u to depth of v
        for k in range(LOG-1, -1, -1):
            if depth[u] - (1 << k) >= depth[v]:
                u = up[k][u]
        if u == v:
            return u
        for k in range(LOG-1, -1, -1):
            if up[k][u] != up[k][v]:
                u = up[k][u]
                v = up[k][v]
        return up[0][u]
    
    # Collect all nodes in A and B
    all_nodes = A + B
    current_lca = all_nodes[0]
    for node in all_nodes[1:]:
        current_lca = lca(current_lca, node)
    root = current_lca
    
    # BFS to compute max_edge_from_root
    max_edge_from_root = [0]*(N+1)
    visited = [False]*(N+1)
    q = deque()
    q.append(root)
    visited[root] = True
    max_edge_from_root[root] = 0
    while q:
        u = q.popleft()
        for v, w in adj[u]:
            if not visited[v]:
                visited[v] = True
                current_max = max(w, max_edge_from_root[u])
                max_edge_from_root[v] = current_max
                q.append(v)
    
    # Get the values for A and B
    a_values = [max_edge_from_root[a] for a in A]
    b_values = [max_edge_from_root[b] for b in B]
    
    a_values.sort()
    b_values.sort()
    
    total = 0
    for a, b in zip(a_values, b_values):
        total += max(a, b)
    print(total)

if __name__ == "__main__":
    main()