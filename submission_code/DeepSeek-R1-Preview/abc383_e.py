import sys
from sys import stdin
from collections import defaultdict

sys.setrecursionlimit(1 << 25)

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx +=1
    M = int(input[idx]); idx +=1
    K = int(input[idx]); idx +=1

    edges = []
    for _ in range(M):
        u = int(input[idx]); idx +=1
        v = int(input[idx]); idx +=1
        w = int(input[idx]); idx +=1
        edges.append((w, u-1, v-1))  # 0-based

    # Kruskal's algorithm to find MST
    edges.sort()
    parent = list(range(N))
    rank = [1]*N

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
        if rank[u_root] < rank[v_root]:
            parent[u_root] = v_root
            rank[v_root] += rank[u_root]
        else:
            parent[v_root] = u_root
            rank[u_root] += v_root
        return True

    mst_edges = []
    for w, u, v in edges:
        if union(u, v):
            mst_edges.append((u, v, w))
            if len(mst_edges) == N-1:
                break

    # Build adjacency list for the MST
    adj = [[] for _ in range(N)]
    for u, v, w in mst_edges:
        adj[u].append((v, w))
        adj[v].append((u, w))

    # Precompute for LCA with binary lifting and max edge
    LOG = 20
    parent_lift = [[-1]*N for _ in range(LOG)]
    max_edge = [[0]*N for _ in range(LOG)]
    depth = [0]*N

    # BFS to set up parent and depth
    from collections import deque
    q = deque()
    root = 0
    q.append(root)
    parent_lift[0][root] = -1
    visited = [False]*N
    visited[root] = True
    while q:
        u = q.popleft()
        for v, w in adj[u]:
            if not visited[v]:
                visited[v] = True
                parent_lift[0][v] = u
                max_edge[0][v] = w
                depth[v] = depth[u] + 1
                q.append(v)

    # Fill the parent_lift and max_edge tables
    for k in range(1, LOG):
        for v in range(N):
            if parent_lift[k-1][v] != -1:
                parent_lift[k][v] = parent_lift[k-1][parent_lift[k-1][v]]
                max_edge[k][v] = max(max_edge[k-1][v], max_edge[k-1][parent_lift[k-1][v]])
            else:
                parent_lift[k][v] = -1
                max_edge[k][v] = 0

    # Function to find LCA and max edge on the path
    def lca(u, v):
        if depth[u] < depth[v]:
            u, v = v, u
        max_w = 0
        # Bring u to the depth of v
        for k in range(LOG-1, -1, -1):
            if depth[u] - (1 << k) >= depth[v]:
                max_w = max(max_w, max_edge[k][u])
                u = parent_lift[k][u]
        if u == v:
            return (u, max_w)
        # Now bring both up until LCA is found
        for k in range(LOG-1, -1, -1):
            if parent_lift[k][u] != -1 and parent_lift[k][u] != parent_lift[k][v]:
                max_w = max(max_w, max_edge[k][u], max_edge[k][v])
                u = parent_lift[k][u]
                v = parent_lift[k][v]
        max_w = max(max_w, max_edge[0][u], max_edge[0][v])
        return (parent_lift[0][u], max_w)

    # Function to compute f(u, v)
    def f(u, v):
        u -=1  # convert to 0-based
        v -=1
        _, w = lca(u, v)
        return w

    # Read A and B
    A = list(map(int, input[idx:idx+K]))
    idx +=K
    B = list(map(int, input[idx:idx+K]))
    idx +=K

    # Compute f for all possible pairs and find the minimal assignment
    # But with K up to 2e5, we need a smarter approach

    # Instead, we'll sort A and B based on their h values (max_edge from root)
    # Compute h for each node
    h = [0]*N
    # BFS from root again to compute h
    visited = [False]*N
    q = deque()
    q.append((root, 0))
    visited[root] = True
    while q:
        u, current_max = q.popleft()
        h[u] = current_max
        for v, w in adj[u]:
            if not visited[v]:
                visited[v] = True
                new_max = max(current_max, w)
                q.append((v, new_max))

    # Now, collect the h values for A and B
    A_h = []
    for a in A:
        node = a-1  # 0-based
        A_h.append(h[node])
    B_h = []
    for b in B:
        node = b-1
        B_h.append(h[node])

    # Sort A and B based on h values
    # We'll sort A and B in non-decreasing order and match
    A_sorted = sorted(A, key=lambda x: h[x-1])
    B_sorted = sorted(B, key=lambda x: h[x-1])

    # Compute sum of f(A_i, B_i) for the sorted pairs
    total =0
    for a, b in zip(A_sorted, B_sorted):
        total += f(a, b)
    print(total)

if __name__ == "__main__":
    main()