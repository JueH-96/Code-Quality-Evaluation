import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    it = iter(data)
    H = int(next(it))
    W = int(next(it))
    n = H * W
    F = []
    for _ in range(H):
        row = list(map(int, [next(it) for _ in range(W)]))
        F.append(row)
    
    edges = []
    for i in range(H):
        for j in range(W):
            idx = i * W + j
            if i < H - 1:
                idx2 = (i + 1) * W + j
                w = min(F[i][j], F[i + 1][j])
                edges.append((w, idx, idx2))
            if j < W - 1:
                idx2 = i * W + (j + 1)
                w = min(F[i][j], F[i][j + 1])
                edges.append((w, idx, idx2))
    
    edges.sort(key=lambda x: x[0], reverse=True)
    
    parent_dsu = list(range(n))
    rank = [0] * n
    
    def find(x):
        if parent_dsu[x] != x:
            parent_dsu[x] = find(parent_dsu[x])
        return parent_dsu[x]
    
    def union(x, y):
        rx = find(x)
        ry = find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            parent_dsu[rx] = ry
        elif rank[rx] > rank[ry]:
            parent_dsu[ry] = rx
        else:
            parent_dsu[ry] = rx
            rank[rx] += 1
        return True
    
    graph = [[] for _ in range(n)]
    for w, u, v in edges:
        if find(u) != find(v):
            union(u, v)
            graph[u].append((v, w))
            graph[v].append((u, w))
    
    depth = [-1] * n
    parent0 = [-1] * n
    edge_weight0 = [0] * n
    depth[0] = 0
    queue = deque([0])
    while queue:
        u = queue.popleft()
        for v, w in graph[u]:
            if depth[v] == -1:
                depth[v] = depth[u] + 1
                parent0[v] = u
                edge_weight0[v] = w
                queue.append(v)
    
    max_k = (n).bit_length()
    dp = [[-1] * n for _ in range(max_k)]
    min_edge_table = [[0] * n for _ in range(max_k)]
    
    for i in range(n):
        dp[0][i] = parent0[i]
        min_edge_table[0][i] = edge_weight0[i]
    
    for k in range(1, max_k):
        for i in range(n):
            if dp[k - 1][i] == -1:
                dp[k][i] = -1
                min_edge_table[k][i] = min_edge_table[k - 1][i]
            else:
                prev_parent = dp[k - 1][i]
                dp[k][i] = dp[k - 1][prev_parent]
                min_edge_table[k][i] = min(min_edge_table[k - 1][i], min_edge_table[k - 1][prev_parent])
    
    def lca_min(u, v):
        if depth[u] < depth[v]:
            u, v = v, u
        min_val = 10**10
        diff = depth[u] - depth[v]
        for k in range(max_k):
            if diff & (1 << k):
                if u == -1:
                    break
                min_val = min(min_val, min_edge_table[k][u])
                u = dp[k][u]
        if u == v:
            return min_val
        for k in range(max_k - 1, -1, -1):
            if dp[k][u] != dp[k][v]:
                min_val = min(min_val, min_edge_table[k][u], min_edge_table[k][v])
                u = dp[k][u]
                v = dp[k][v]
        min_val = min(min_val, min_edge_table[0][u], min_edge_table[0][v])
        return min_val
    
    Q = int(next(it))
    out_lines = []
    for _ in range(Q):
        A = int(next(it))
        B = int(next(it))
        Y = int(next(it))
        C = int(next(it))
        D = int(next(it))
        Z = int(next(it))
        if (A - 1, B - 1) == (C - 1, D - 1):
            ans = abs(Y - Z)
            out_lines.append(str(ans))
        else:
            u = (A - 1) * W + (B - 1)
            v = (C - 1) * W + (D - 1)
            M0 = lca_min(u, v)
            low = min(Y, Z)
            if M0 >= low:
                ans = abs(Y - Z)
            else:
                ans = Y + Z - 2 * M0
            out_lines.append(str(ans))
    
    sys.stdout.write("
".join(out_lines) + "
")

if __name__ == "__main__":
    main()