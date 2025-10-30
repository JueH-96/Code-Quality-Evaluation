import sys
sys.setrecursionlimit(10**6)
input = sys.stdin.readline

def main():
    data = input().split()
    if not data:
        return
    H = int(data[0]); W = int(data[1])
    grid = []
    for i in range(H):
        row = list(map(int, input().split()))
        grid.append(row)
    
    n = H * W
    edges = []
    for i in range(H):
        for j in range(W):
            if i < H-1:
                w = min(grid[i][j], grid[i+1][j])
                u = i * W + j
                v = (i+1) * W + j
                edges.append((w, u, v))
            if j < W-1:
                w = min(grid[i][j], grid[i][j+1])
                u = i * W + j
                v = i * W + j+1
                edges.append((w, u, v))
                
    edges.sort(key=lambda x: x[0], reverse=True)
    
    parent_uf = list(range(n))
    rank = [0] * n
    
    def find(x):
        if parent_uf[x] != x:
            parent_uf[x] = find(parent_uf[x])
        return parent_uf[x]
    
    def union(x, y):
        rx = find(x)
        ry = find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            parent_uf[rx] = ry
        elif rank[rx] > rank[ry]:
            parent_uf[ry] = rx
        else:
            parent_uf[ry] = rx
            rank[rx] += 1
        return True
            
    adj = [[] for _ in range(n)]
    count_edges = 0
    for w, u, v in edges:
        if union(u, v):
            adj[u].append((v, w))
            adj[v].append((u, w))
            count_edges += 1
            if count_edges == n-1:
                break
                
    depth = [-1] * n
    parent0 = [-1] * n
    min_edge0 = [0] * n
    
    stack = [0]
    depth[0] = 0
    parent0[0] = -1
    min_edge0[0] = 10**18
    
    while stack:
        u = stack.pop()
        for (v, w) in adj[u]:
            if v == parent0[u]:
                continue
            parent0[v] = u
            min_edge0[v] = w
            depth[v] = depth[u] + 1
            stack.append(v)
            
    LOG = (n).bit_length()
    parent_tab = [[-1] * n for _ in range(LOG)]
    min_edge_tab = [[10**18] * n for _ in range(LOG)]
    
    for i in range(n):
        parent_tab[0][i] = parent0[i]
        if parent0[i] != -1:
            min_edge_tab[0][i] = min_edge0[i]
        else:
            min_edge_tab[0][i] = 10**18
            
    for k in range(1, LOG):
        for i in range(n):
            p = parent_tab[k-1][i]
            if p == -1:
                parent_tab[k][i] = -1
                min_edge_tab[k][i] = min_edge_tab[k-1][i]
            else:
                parent_tab[k][i] = parent_tab[k-1][p]
                min_edge_tab[k][i] = min(min_edge_tab[k-1][i], min_edge_tab[k-1][p])
                
    def lca_min(u, v):
        if depth[u] < depth[v]:
            u, v = v, u
        min_val = 10**18
        d = depth[u] - depth[v]
        k = 0
        while d:
            if d & 1:
                min_val = min(min_val, min_edge_tab[k][u])
                u = parent_tab[k][u]
            d >>= 1
            k += 1
            
        if u == v:
            return min_val
            
        for k in range(LOG-1, -1, -1):
            if parent_tab[k][u] != parent_tab[k][v]:
                min_val = min(min_val, min_edge_tab[k][u], min_edge_tab[k][v])
                u = parent_tab[k][u]
                v = parent_tab[k][v]
        min_val = min(min_val, min_edge_tab[0][u], min_edge_tab[0][v])
        return min_val
        
    Q = int(input().strip())
    out_lines = []
    for _ in range(Q):
        data = list(map(int, input().split()))
        A = data[0]-1; B = data[1]-1; Y_i = data[2]
        C = data[3]-1; D = data[4]-1; Z_i = data[5]
        u_idx = A * W + B
        v_idx = C * W + D
        
        if u_idx == v_idx:
            ans = abs(Y_i - Z_i)
            out_lines.append(str(ans))
            continue
            
        X = lca_min(u_idx, v_idx)
        a_val = Y_i
        b_val = Z_i
        L = min(a_val, b_val)
        if X >= L:
            ans = abs(a_val - b_val)
        else:
            ans = a_val + b_val - 2 * X
        out_lines.append(str(ans))
        
    print("
".join(out_lines))

if __name__ == "__main__":
    main()