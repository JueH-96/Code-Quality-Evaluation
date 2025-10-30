# YOUR CODE HERE
def main():
    import sys,sys
    sys.setrecursionlimit(10**6)
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    try:
        H = int(next(it))
    except StopIteration:
        return
    W = int(next(it))
    n = H * W
    F = [0] * n
    # Read the grid (row‐by‐row; we use 0-based indexing)
    for i in range(H):
        for j in range(W):
            F[i * W + j] = int(next(it))
            
    # Build the edge list.
    # For each cell, add an edge to its right neighbour and its down neighbour (if within bounds).
    edges = []  # Each edge is a tuple (w, u, v) with w = min(F[u], F[v])
    for i in range(H):
        base = i * W
        for j in range(W):
            u = base + j
            if j + 1 < W:
                v = u + 1
                # weight = min(F[u], F[v])
                if F[u] < F[v]:
                    w = F[u]
                else:
                    w = F[v]
                edges.append((w, u, v))
            if i + 1 < H:
                v = u + W
                if F[u] < F[v]:
                    w = F[u]
                else:
                    w = F[v]
                edges.append((w, u, v))
    # sort edges descending (largest weight first)
    edges.sort(key=lambda x: x[0], reverse=True)
    
    # DSU union–find to build the maximum spanning tree.
    parent = list(range(n))
    rank = [0] * n

    def dsu_find(x):
        # iterative find with path halving
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def dsu_union(a, b):
        ra = dsu_find(a)
        rb = dsu_find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    # Build MST (maximum spanning tree) using Kruskal.
    mst = [[] for _ in range(n)]
    for w, u, v in edges:
        if dsu_find(u) != dsu_find(v):
            dsu_union(u, v)
            mst[u].append((v, w))
            mst[v].append((u, w))
    # Now mst is a spanning tree of the grid.
    
    # Do a DFS/BFS to compute the parent (par), depth and min_edge from each node to its parent.
    depth = [-1] * n
    par = [-1] * n
    min_edge_to_parent = [10**9] * n  # for the root we use a large value.
    stack = [0]  # choose node 0 as the root.
    depth[0] = 0
    par[0] = -1
    min_edge_to_parent[0] = 10**9
    while stack:
        u = stack.pop()
        for (v, w) in mst[u]:
            if depth[v] == -1:
                depth[v] = depth[u] + 1
                par[v] = u
                min_edge_to_parent[v] = w
                stack.append(v)
                
    # Binary lifting: build "up" and "min_up" tables.
    L = (n).bit_length()  # number of levels; for n<=250k, L is about 19.
    up = []
    up.append(par[:])
    INF = 10**9
    min_up = []
    min_up.append(min_edge_to_parent[:])
    for k in range(1, L):
        prev = up[k-1]
        curr = [-1] * n
        curr_min = [INF] * n
        for v in range(n):
            pv = prev[v]
            if pv == -1:
                curr[v] = -1
                curr_min[v] = INF
            else:
                curr[v] = prev[pv]
                a = min_up[k-1][v]
                b = min_up[k-1][pv]
                curr_min[v] = a if a < b else b
        up.append(curr)
        min_up.append(curr_min)
    
    # A query on the MST: for nodes u and v, we want the minimum over edge weights along the path
    # (which equals M(u,v)). If u==v, we return F[u].
    def query_min_edge(u, v):
        if u == v:
            return F[u]
        ret = INF
        if depth[u] < depth[v]:
            u, v = v, u
        diff = depth[u] - depth[v]
        bit = 0
        while diff:
            if diff & 1:
                if min_up[bit][u] < ret:
                    ret = min_up[bit][u]
                u = up[bit][u]
            diff //= 2
            bit += 1
        if u == v:
            return ret
        for k in range(L-1, -1, -1):
            if up[k][u] != up[k][v]:
                if min_up[k][u] < ret:
                    ret = min_up[k][u]
                if min_up[k][v] < ret:
                    ret = min_up[k][v]
                u = up[k][u]
                v = up[k][v]
        if min_up[0][u] < ret:
            ret = min_up[0][u]
        if min_up[0][v] < ret:
            ret = min_up[0][v]
        return ret

    # Process queries.
    Q = int(next(it))
    out_lines = []
    # (Recall: answer = |Y - Z| if (Y<=M or Z<=M), else (Y-M)+(Z-M) )
    for _ in range(Q):
        A = int(next(it))
        B = int(next(it))
        Y_val = int(next(it))
        C = int(next(it))
        D = int(next(it))
        Z_val = int(next(it))
        u = (A - 1) * W + (B - 1)
        v = (C - 1) * W + (D - 1)
        if u == v:
            M_val = F[u]
        else:
            M_val = query_min_edge(u, v)
        if Y_val > M_val and Z_val > M_val:
            ans = (Y_val - M_val) + (Z_val - M_val)
        else:
            ans = Y_val - Z_val if Y_val >= Z_val else Z_val - Y_val
        out_lines.append(str(ans))
    sys.stdout.write("
".join(out_lines))

if __name__ == '__main__':
    main()