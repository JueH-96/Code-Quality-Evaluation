import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())

    # adj_matrix: list of N integers, where each integer represents a row
    # The i-th bit of adj_matrix[row] is 1 if (row, i) is an edge.
    adj_matrix = [0] * N 

    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        u -= 1 # Convert to 0-indexed
        v -= 1
        adj_matrix[u] |= (1 << v)
        adj_matrix[v] |= (1 << u)

    # Make a copy for Gaussian elimination as the original `adj_matrix` might be needed
    # (though not in this specific logic, it's good practice)
    temp_matrix = list(adj_matrix) 

    # pivot_col_map: maps current_row_idx to its pivot_col_idx in the RREF
    pivot_col_map = {}
    
    current_row = 0
    for col in range(N):
        # Find a suitable pivot row for the current column
        pivot_row_idx = current_row
        while pivot_row_idx < N and not ((temp_matrix[pivot_row_idx] >> col) & 1):
            pivot_row_idx += 1
        
        if pivot_row_idx < N: # A pivot (1) is found in `col`
            # Swap current_row with pivot_row_idx to bring the pivot to `current_row`
            temp_matrix[current_row], temp_matrix[pivot_row_idx] = temp_matrix[pivot_row_idx], temp_matrix[current_row]
            
            # Record the pivot column for this row
            pivot_col_map[current_row] = col
            
            # Eliminate 1s in `col` from all other rows
            for r in range(N):
                if r != current_row and ((temp_matrix[r] >> col) & 1):
                    temp_matrix[r] ^= temp_matrix[current_row]
            current_row += 1
    
    # The rank of the matrix is the number of pivot rows found
    rank = current_row
    
    # If rank equals N, the null space only contains the zero vector.
    # This means all b_v,k must be 0, leading to X_v = 0 for all v, which is not allowed.
    if rank == N:
        print("No")
        return

    # Construct a basis for the null space
    # The number of basis vectors (dimension of null space) is N - rank
    
    # Identify pivot columns and free columns
    pivot_cols = set(pivot_col_map.values())
    free_cols = sorted(list(set(range(N)) - pivot_cols))
    
    basis_vectors = [] # Each element is an N-bit integer representing a basis vector
    
    # For each free variable, construct a basis vector
    for free_col_idx in free_cols:
        basis_vector_bits = [0] * N
        basis_vector_bits[free_col_idx] = 1 # Set the current free variable to 1
        
        # Determine values for pivot variables based on the RREF
        for r_idx in range(rank): # Iterate through rows that had pivots
            p_col_idx = pivot_col_map[r_idx] # The pivot column for this row
            
            # The equation for this row (in RREF) is:
            # 1 * b[p_col_idx] + sum_{k in free_cols} (temp_matrix[r_idx]>>k)&1 * b[k] = 0 (mod 2)
            # So, b[p_col_idx] = sum_{k in free_cols} ((temp_matrix[r_idx]>>k)&1) * b[k] (mod 2)
            
            val_for_pivot_var = 0
            for k_idx in free_cols:
                if ((temp_matrix[r_idx] >> k_idx) & 1) and (basis_vector_bits[k_idx] == 1):
                    val_for_pivot_var ^= 1
            basis_vector_bits[p_col_idx] = val_for_pivot_var
        
        # Convert the list of bits (representing the basis vector) to an integer
        # Note: bit manipulation often works with LSB at index 0. Reversing `basis_vector_bits` ensures
        # that `int("".join(map(str, ...)), 2)` treats `basis_vector_bits[0]` as the LSB.
        basis_vectors.append(int("".join(map(str, basis_vector_bits[::-1])), 2))

    # Check the condition: for every vertex v, its profile vector P_v must be non-zero.
    # P_v is formed by the v-th bit of each basis vector (B_1,v, B_2,v, ..., B_k,v).
    # If P_v is all zeros, then X_v would be 0, which is not allowed.
    
    num_basis_vectors = len(basis_vectors) # This is equal to N - rank
    
    is_solution_possible = True
    for v_idx in range(N):
        pv_is_zero = True
        for j in range(num_basis_vectors):
            if ((basis_vectors[j] >> v_idx) & 1): # Check the v-th bit of the j-th basis vector
                pv_is_zero = False
                break
        if pv_is_zero:
            is_solution_possible = False
            break
            
    if not is_solution_possible:
        print("No")
    else:
        print("Yes")
        # Construct X_v for each vertex v.
        # We choose the coefficients c_j,k such that the k-th bit of X_v
        # is the (v-th bit of basis_vectors[k]) for k < num_basis_vectors, and 0 otherwise.
        # This guarantees X_v >= 1 because P_v is non-zero.
        
        result_X = [0] * N
        for v_idx in range(N):
            val_Xv = 0
            for k in range(num_basis_vectors):
                # The k-th bit of X_v is b_{v,k}. We set b_{v,k} to be P_{v,k},
                # which is the v-th bit of the k-th basis vector (basis_vectors[k]).
                if ((basis_vectors[k] >> v_idx) & 1):
                    val_Xv |= (1 << k) # Set the k-th bit of X_v
            result_X[v_idx] = val_Xv
        
        # Print the resulting X_v values, space-separated
        print(*(x for x in result_X))

solve()