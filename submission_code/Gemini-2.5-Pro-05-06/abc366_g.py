import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    
    if M == 0:
        # All vertices isolated. No conditions for XOR sums.
        # Assign X_v = 1 for all v.
        print("Yes")
        print(*( [1] * N ))
        return

    # mat is the matrix for the system of equations Ax = 0
    # where A is the adjacency matrix of the graph.
    mat = [[0] * N for _ in range(N)]
    
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        u -= 1 # 0-indexed
        v -= 1 # 0-indexed
        mat[u][v] = 1
        mat[v][u] = 1
        
    # Gaussian elimination to find RRE form (over GF(2))
    pivot_cols = [] # Stores column indices of pivots
    
    curr_pivot_row = 0
    for pcol in range(N): # Iterate through potential pivot columns
        if curr_pivot_row == N: 
            break
            
        pivot_r_candidate = curr_pivot_row
        while pivot_r_candidate < N and mat[pivot_r_candidate][pcol] == 0:
            pivot_r_candidate += 1
        
        if pivot_r_candidate < N: # Found a pivot for pcol
            mat[curr_pivot_row], mat[pivot_r_candidate] = mat[pivot_r_candidate], mat[curr_pivot_row]
            
            for r_idx in range(N):
                if r_idx != curr_pivot_row and mat[r_idx][pcol] == 1:
                    for c_idx in range(N): # Iterate over all columns for XOR sum
                        mat[r_idx][c_idx] ^= mat[curr_pivot_row][c_idx]
            
            pivot_cols.append(pcol)
            curr_pivot_row += 1
            
    rank = len(pivot_cols)
    
    all_cols_indices = set(range(N))
    pivot_cols_set = set(pivot_cols)
    free_cols = sorted(list(all_cols_indices - pivot_cols_set))
    
    num_free_vars = len(free_cols)

    if num_free_vars == 0:
        print("No")
        return

    basis_vectors = []
    for k_free_var_idx in range(num_free_vars):
        current_free_col_idx = free_cols[k_free_var_idx]
        
        bv = [0] * N
        bv[current_free_col_idx] = 1
        
        for r_in_rre in range(rank): 
            pcol_of_this_row = pivot_cols[r_in_rre] # Column where mat[r_in_rre] has its pivot
            if mat[r_in_rre][current_free_col_idx] == 1:
                bv[pcol_of_this_row] = 1
        
        basis_vectors.append(bv)

    for j_var_idx in range(N):
        is_settable_to_one = False
        for k_basis_idx in range(num_free_vars):
            if basis_vectors[k_basis_idx][j_var_idx] == 1:
                is_settable_to_one = True
                break
        if not is_settable_to_one:
            print("No")
            return
            
    X_values = [0] * N
    for j_var_idx in range(N):
        val = 0
        for k_basis_idx in range(num_free_vars):
            if basis_vectors[k_basis_idx][j_var_idx] == 1:
                val ^= (1 << k_basis_idx)
        X_values[j_var_idx] = val
    
    print("Yes")
    print(*(x for x in X_values))

solve()