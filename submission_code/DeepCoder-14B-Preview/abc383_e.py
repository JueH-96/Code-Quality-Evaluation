import sys
from collections import defaultdict, deque

sys.setrecursionlimit(1 << 25)

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr += 1
    M = int(input[ptr]); ptr += 1
    K = int(input[ptr]); ptr += 1

    edges = []
    for _ in range(M):
        u = int(input[ptr]); ptr += 1
        v = int(input[ptr]); ptr += 1
        w = int(input[ptr]); ptr += 1
        edges.append((w, u-1, v-1))  # zero-based

    # Sort edges for Kruskal's algorithm
    edges.sort()

    # Kruskal's algorithm to build MST
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

    mst_edges = []
    for w, u, v in edges:
        if union(u, v):
            mst_edges.append((u, v, w))
            if len(mst_edges) == N - 1:
                break

    # Build adjacency list for MST
    adj = [[] for _ in range(N)]
    for u, v, w in mst_edges:
        adj[u].append((v, w))
        adj[v].append((u, w))

    # Preprocess for LCA and maximum edge on path
    LOG = 20
    depth = [0] * N
    up = [[-1] * N for _ in range(LOG)]
    max_edge = [[0] * N for _ in range(LOG)]

    # BFS to set up the tree and depth
    visited = [False] * N
    q = deque()
    q.append(0)
    visited[0] = True
    up[0][0] = -1
    while q:
        u = q.popleft()
        for v, w in adj[u]:
            if not visited[v] and v != up[0][u]:
                visited[v] = True
                up[0][v] = u
                depth[v] = depth[u] + 1
                max_edge[0][v] = w
                q.append(v)

    # Fill the up and max_edge tables
    for k in range(1, LOG):
        for v in range(N):
            up_k_minus_1 = up[k-1][v]
            if up_k_minus_1 != -1:
                up[k][v] = up[k-1][up_k_minus_1]
                max_edge[k][v] = max(max_edge[k-1][v], max_edge[k-1][up_k_minus_1])
            else:
                up[k][v] = -1
                max_edge[k][v] = 0

    def lca(u, v):
        if depth[u] < depth[v]:
            u, v = v, u
        # Bring u to the level of v
        for k in range(LOG-1, -1, -1):
            if depth[u] - (1 << k) >= depth[v]:
                u = up[k][u]
        if u == v:
            return u
        for k in range(LOG-1, -1, -1):
            if up[k][u] != -1 and up[k][u] != up[k][v]:
                u = up[k][u]
                v = up[k][v]
        return up[0][u]

    def get_max_edge(u, v):
        ancestor = lca(u, v)
        max_e = 0
        # Move u up to ancestor
        for k in range(LOG-1, -1, -1):
            if depth[u] - (1 << k) >= depth[ancestor]:
                max_e = max(max_e, max_edge[k][u])
                u = up[k][u]
        # Move v up to ancestor
        for k in range(LOG-1, -1, -1):
            if depth[v] - (1 << k) >= depth[ancestor]:
                max_e = max(max_e, max_edge[k][v])
                v = up[k][v]
        return max_e

    # Read A and B
    A = list(map(int, input[ptr:ptr+K]))
    ptr += K
    B = list(map(int, input[ptr:ptr+K]))
    ptr += K

    # Convert to zero-based
    A = [x-1 for x in A]
    B = [x-1 for x in B]

    # Sort A and B
    A_sorted = sorted(A)
    B_sorted = sorted(B)

    total = 0
    for a, b in zip(A_sorted, B_sorted):
        total += get_max_edge(a, b)

    print(total)

if __name__ == '__main__':
    main()