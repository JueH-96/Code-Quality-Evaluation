# YOUR CODE HERE
import sys

# It's a graph problem on a large grid, which might lead to deep recursion.
# Set a higher recursion limit.
sys.setrecursionlimit(3 * 10**5)

def main():
    """
    This function implements the solution described in the explanation.
    It first preprocesses the grid to answer queries efficiently, then
    iterates through each query to calculate and print the result.
    """
    try:
        # Fast I/O
        readline = sys.stdin.readline
        
        H, W = map(int, readline().split())
        F = [list(map(int, readline().split())) for _ in range(H)]
        Q = int(readline())
        queries = [list(map(int, readline().split())) for _ in range(Q)]
    except (IOError, ValueError):
        # Handle potential empty input on some platforms
        return

    N = H * W
    if N == 0:
        return
    LOG_N = (N - 1).bit_length()

    def node_id(r, c):
        return r * W + c

    edges = []
    for r in range(H):
        for c in range(W):
            if r + 1 < H:
                w = min(F[r][c], F[r + 1][c])
                edges.append((w, node_id(r, c), node_id(r + 1, c)))
            if c + 1 < W:
                w = min(F[r][c], F[r][c + 1])
                edges.append((w, node_id(r, c), node_id(r, c + 1)))

    edges.sort(key=lambda x: x[0], reverse=True)

    # DSU for Kruskal's
    parent_dsu = list(range(N))
    def find_set(v):
        # Iterative find with path compression
        path_to_root = []
        while v != parent_dsu[v]:
            path_to_root.append(v)
            v = parent_dsu[v]
        for node in path_to_root:
            parent_dsu[node] = v
        return v

    def unite_sets(a, b):
        a = find_set(a)
        b = find_set(b)
        if a != b:
            parent_dsu[b] = a
    
    # Build Maximum Spanning Forest
    mst_adj = [[] for _ in range(N)]
    for w, u, v in edges:
        if find_set(u) != find_set(v):
            unite_sets(u, v)
            mst_adj[u].append((v, w))
            mst_adj[v].append((u, w))

    # Precomputation for LCA
    depth = [-1] * N
    lca_parent = [[-1] * LOG_N for _ in range(N)]
    min_weight = [[0] * LOG_N for _ in range(N)]

    # DFS to build parent and depth arrays for all components of the forest
    for i in range(N):
        if depth[i] == -1:
            stack = [(i, -1, 0, float('inf'))]  # (u, p, d, w_to_p)
            while stack:
                u, p, d, w_edge = stack.pop()
                depth[u] = d
                lca_parent[u][0] = p
                min_weight[u][0] = w_edge
                for v, w in mst_adj[u]:
                    if v != p:
                        stack.append((v, u, d + 1, w))

    # Binary lifting tables
    for k in range(1, LOG_N):
        for i in range(N):
            p = lca_parent[i][k - 1]
            if p != -1:
                lca_parent[i][k] = lca_parent[p][k - 1]
                min_weight[i][k] = min(min_weight[i][k - 1], min_weight[p][k - 1])

    # For final DSU check
    final_roots = [find_set(i) for i in range(N)]

    def get_f_conn(u, v):
        if final_roots[u] != final_roots[v]:
            return 0
        if u == v:
            return float('inf')

        res = float('inf')
        
        if depth[u] < depth[v]:
            u, v = v, u

        for k in range(LOG_N - 1, -1, -1):
            if lca_parent[u][k] != -1 and depth[u] - (1 << k) >= depth[v]:
                res = min(res, min_weight[u][k])
                u = lca_parent[u][k]

        if u == v:
            return res

        for k in range(LOG_N - 1, -1, -1):
            if lca_parent[u][k] != -1 and lca_parent[u][k] != lca_parent[v][k]:
                res = min(res, min_weight[u][k], min_weight[v][k])
                u = lca_parent[u][k]
                v = lca_parent[v][k]
        
        res = min(res, min_weight[u][0], min_weight[v][0])
        return res

    # Process queries
    for A, B, Y, C, D, Z in queries:
        u = node_id(A - 1, B - 1)
        v = node_id(C - 1, D - 1)
        
        f_conn = get_f_conn(u, v)
        
        y_min = min(Y, Z)
        
        if f_conn >= y_min:
            ans = abs(Y - Z)
        else:
            ans = (Y - f_conn) + (Z - f_conn)
            
        print(ans)

if __name__ == "__main__":
    main()