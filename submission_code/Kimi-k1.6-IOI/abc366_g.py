def gaussian_elimination(rows, N):
    matrix = [row.copy() for row in rows]
    n_rows = len(matrix)
    rank = 0
    pivots = [-1] * n_rows

    # Forward elimination
    for col in range(N):
        pivot_row = -1
        for r in range(rank, n_rows):
            if matrix[r][col]:
                pivot_row = r
                break
        if pivot_row == -1:
            continue
        matrix[rank], matrix[pivot_row] = matrix[pivot_row], matrix[rank]
        pivots[rank] = col
        for r in range(rank + 1, n_rows):
            if matrix[r][col]:
                matrix[r] = [(matrix[r][c] ^ matrix[rank][c]) for c in range(N)]
        rank += 1

    # Backward elimination
    for r in range(rank - 1, -1, -1):
        p_col = pivots[r]
        for u in range(r - 1, -1, -1):
            if matrix[u][p_col]:
                matrix[u] = [matrix[u][c] ^ matrix[r][c] for c in range(N)]

    # Generate basis vectors
    basis = []
    pivot_cols = set(pivots[:rank])
    for j in range(N):
        if j not in pivot_cols:
            vec = [0] * N
            vec[j] = 1
            for r in range(rank):
                p_col = pivots[r]
                vec[p_col] = matrix[r][j]
            basis.append(vec)
    return basis, rank

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx +=1
    M = int(input[idx])
    idx +=1
    
    edges = [[] for _ in range(N)]
    degrees = [0]*N
    for _ in range(M):
        u = int(input[idx])-1
        v = int(input[idx+1])-1
        idx +=2
        edges[u].append(v)
        edges[v].append(u)
        degrees[u] +=1
        degrees[v] +=1
    
    rows_vertices = [v for v in range(N) if degrees[v] >=1]
    matrix = []
    for v in rows_vertices:
        row = [0]*N
        for u in edges[v]:
            row[u] = 1
        matrix.append(row)
    
    if not matrix:
        print("Yes")
        print(' '.join(map(str, [1]*N)))
        return
    
    basis, rank_A = gaussian_elimination(matrix, N)
    
    if rank_A >= N:
        print("No")
        return
    
    variables_with_degree = [v for v in range(N) if degrees[v] >=1]
    for v in variables_with_degree:
        covered = False
        for vec in basis:
            if vec[v]:
                covered = True
                break
        if not covered:
            print("No")
            return
    
    X = [0] * N
    bit = 0
    for vec in basis:
        for v in range(N):
            if vec[v]:
                X[v] |= (1 << bit)
        bit +=1
        if bit >= 60:
            break  # Avoid exceeding bit limit, though N <=60 ensures this won't happen
    
    for v in range(N):
        if degrees[v] ==0:
            X[v] |=1
    
    print("Yes")
    print(' '.join(map(str, X)))

if __name__ == "__main__":
    main()