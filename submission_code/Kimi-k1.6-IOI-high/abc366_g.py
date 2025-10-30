def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    
    adj = [[] for _ in range(N + 1)]  # 1-based indexing
    degree = [0] * (N + 1)
    for _ in range(M):
        u = int(input[ptr])
        ptr += 1
        v = int(input[ptr])
        ptr += 1
        adj[u].append(v)
        adj[v].append(u)
        degree[u] += 1
        degree[v] += 1
    
    S_nodes = [v for v in range(1, N + 1) if degree[v] >= 1]
    
    if not S_nodes:
        print("Yes")
        print(' '.join('1' for _ in range(N)))
        return
    
    # Build matrix rows
    rows = []
    for v in S_nodes:
        bits = 0
        for u in adj[v]:
            bits ^= (1 << (u - 1))  # u is 1-based to 0-based index
        rows.append(bits)
    
    col_count = N  # 0-based columns
    matrix = rows.copy()
    row_rank = 0
    pivots = []
    n_rows = len(matrix)
    
    # Perform Gaussian Elimination
    for col in range(col_count):
        if row_rank >= n_rows:
            break
        pivot_row = -1
        for r in range(row_rank, n_rows):
            if (matrix[r] >> col) & 1:
                pivot_row = r
                break
        if pivot_row == -1:
            continue
        # Swap current row_rank with pivot_row
        matrix[row_rank], matrix[pivot_row] = matrix[pivot_row], matrix[row_rank]
        pivots.append(col)
        # Eliminate other rows
        for r in range(n_rows):
            if r != row_rank and (matrix[r] >> col) & 1:
                matrix[r] ^= matrix[row_rank]
        row_rank += 1
    
    # Generate basis vectors
    free_vars = [col for col in range(col_count) if col not in pivots]
    basis_vectors = []
    for fv in free_vars:
        vec = (1 << fv)
        for i in range(len(pivots)):
            p = pivots[i]
            if (matrix[i] >> fv) & 1:
                vec |= (1 << p)
        basis_vectors.append(vec)
    
    # Check if all nodes in S_nodes are covered
    viable = True
    for v in S_nodes:
        v_idx = v - 1
        found = False
        for bv in basis_vectors:
            if (bv >> v_idx) & 1:
                found = True
                break
        if not found:
            viable = False
            break
    
    if not viable:
        print("No")
        return
    
    # Assign bits and generate X values
    X = [0] * (N + 1)  # 1-based to 0-based nodes (1 to N)
    num_basis = len(basis_vectors)
    for i in range(num_basis):
        bit_position = 59 - i
        vec = basis_vectors[i]
        for v in range(1, N + 1):
            v_idx = v - 1
            if (vec >> v_idx) & 1:
                X[v] |= (1 << bit_position)
    
    # Handle nodes with degree zero
    for v in range(1, N + 1):
        if degree[v] == 0:
            X[v] = 1
    
    # Ensure all X values are non-zero
    for v in range(1, N + 1):
        if X[v] == 0:
            X[v] = 1
    
    print("Yes")
    print(' '.join(map(str, X[1:N+1])))

if __name__ == "__main__":
    main()