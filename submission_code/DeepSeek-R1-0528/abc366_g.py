def gf2_rank(matrix):
    if len(matrix) == 0:
        return 0
    n_rows = len(matrix)
    n_cols = len(matrix[0])
    mat = [row[:] for row in matrix]
    rank = 0
    for j in range(n_cols):
        pivot = -1
        for i in range(rank, n_rows):
            if mat[i][j] == 1:
                pivot = i
                break
        if pivot == -1:
            continue
        mat[rank], mat[pivot] = mat[pivot], mat[rank]
        for i in range(rank + 1, n_rows):
            if mat[i][j] == 1:
                for k in range(j, n_cols):
                    mat[i][k] ^= mat[rank][k]
        rank += 1
    return rank

def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    m = int(data[1])
    edges = []
    index = 2
    graph = [[] for _ in range(n + 1)]
    deg = [0] * (n + 1)
    
    for i in range(m):
        u = int(data[index])
        v = int(data[index + 1])
        index += 2
        edges.append((u, v))
        graph[u].append(v)
        graph[v].append(u)
        deg[u] += 1
        deg[v] += 1
        
    V0 = [v for v in range(1, n + 1) if deg[v] > 0]
    isolated = [v for v in range(1, n + 1) if deg[v] == 0]
    
    if len(V0) == 0:
        print("Yes")
        print(" ".join(["1"] * n))
        return
        
    n0 = len(V0)
    idx_map = {}
    for idx, v in enumerate(V0):
        idx_map[v] = idx
        
    M = [[0] * n0 for _ in range(n0)]
    for i, v in enumerate(V0):
        for neighbor in graph[v]:
            if deg[neighbor] > 0:
                j = idx_map[neighbor]
                M[i][j] = 1
                
    r0 = gf2_rank(M)
    
    for i in range(n0):
        e_i = [0] * n0
        e_i[i] = 1
        new_M = [row[:] for row in M]
        new_M.append(e_i)
        r1 = gf2_rank(new_M)
        if r1 == r0:
            print("No")
            return
            
    mat = [row[:] for row in M]
    n_rows = n0
    n_cols = n0
    pivot_cols = []
    for j in range(n_cols):
        pivot = -1
        for i in range(len(pivot_cols), n_rows):
            if mat[i][j] == 1:
                pivot = i
                break
        if pivot == -1:
            continue
        pivot_cols.append(j)
        if pivot != len(pivot_cols) - 1:
            mat[len(pivot_cols) - 1], mat[pivot] = mat[pivot], mat[len(pivot_cols) - 1]
        pivot_row = len(pivot_cols) - 1
        for i in range(n_rows):
            if i != pivot_row and mat[i][j] == 1:
                for k in range(j, n_cols):
                    mat[i][k] ^= mat[pivot_row][k]
                    
    free_cols = [j for j in range(n_cols) if j not in pivot_cols]
    k = len(free_cols)
    
    basis = []
    for j in free_cols:
        vec = [0] * n_cols
        vec[j] = 1
        for i in range(len(pivot_cols) - 1, -1, -1):
            col = pivot_cols[i]
            s = 0
            for k in range(col + 1, n_cols):
                s ^= mat[i][k] * vec[k]
            vec[col] = s
        basis.append(vec)
        
    ans_arr = [0] * (n + 1)
    for i in range(1, n + 1):
        if deg[i] == 0:
            ans_arr[i] = 1
            
    for v in V0:
        idx = idx_map[v]
        base_value = 0
        for i in range(k):
            if basis[i][idx]:
                base_value |= (1 << i)
        if k <= 58:
            base_value <<= 2
        ans_arr[v] = base_value
        
    print("Yes")
    output_list = [str(ans_arr[i]) for i in range(1, n + 1)]
    print(" ".join(output_list))

if __name__ == "__main__":
    main()