import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    
    # Adjacency matrix represented by rows as integers.
    # Each row A[r] is an integer where the c-th bit is 1 if (r, c) is an edge.
    A = [0] * N
    is_any_edge = False
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        u -= 1 # Convert to 0-based index
        v -= 1 # Convert to 0-based index
        # Set bits corresponding to neighbors
        A[u] |= (1 << v)
        A[v] |= (1 << u)
        is_any_edge = True # Flag if graph has any edges

    # Perform Gaussian elimination over GF(2) to find the Reduced Row Echelon Form (RREF) of A.
    # A_rref will be modified in-place.
    A_rref = list(A) 
    rank = 0
    # pivot_cols[r] stores the column index of the pivot in row r (-1 if row r has no pivot).
    pivot_cols = [-1] * N 
    
    # Map from pivot column index to the row index containing that pivot.
    pivot_row_indices = {} 

    row = 0
    # Forward elimination phase: Transform matrix into Row Echelon Form.
    for col in range(N):
        # If we have processed N rows, we are done with finding pivots.
        if row >= N: break
        
        # Find a row >= `row` with a 1 in column `col` (pivot element).
        pivot_r = row
        while pivot_r < N and (A_rref[pivot_r] >> col) & 1 == 0:
            pivot_r += 1
        
        if pivot_r < N: # Found a pivot at (pivot_r, col)
            # Swap row `row` and `pivot_r` to bring pivot to current row `row`.
            A_rref[row], A_rref[pivot_r] = A_rref[pivot_r], A_rref[row]
            
            # The value of the pivot row. We use this to eliminate 1s in the same column.
            pivot_row_val = A_rref[row]
            
            # Eliminate 1s in column `col` below the pivot row `row`.
            # XORing the pivot row with other rows having 1 at `col` achieves this.
            for i in range(row + 1, N):
                if (A_rref[i] >> col) & 1:
                    A_rref[i] ^= pivot_row_val
            
            # Record the pivot position: column `col` has pivot in row `row`.
            pivot_row_indices[col] = row 
            row += 1 # Move to the next row for finding next pivot.

    # Rank of the matrix is the number of pivot rows found.
    rank = row 

    # Back substitution phase: Transform Row Echelon Form into RREF.
    # Iterate through pivot columns found in reverse order.
    pivot_cols_list = sorted(pivot_row_indices.keys(), reverse=True)

    for col in pivot_cols_list:
        pivot_r = pivot_row_indices[col] # Row index containing the pivot for this column.
        pivot_row_val = A_rref[pivot_r]  # Value of the pivot row.
        
        # Eliminate 1s in column `col` above the pivot row `pivot_r`.
        # XORing the pivot row with rows above it having 1 at `col` achieves this.
        for i in range(pivot_r):
             if (A_rref[i] >> col) & 1:
                 A_rref[i] ^= pivot_row_val

    # Rebuild pivot_cols array maping row index to pivot column index for easier access later.
    basic_vars_cols = set() # Set of column indices corresponding to basic variables (pivot columns).
    for col, r in pivot_row_indices.items():
        pivot_cols[r] = col
        basic_vars_cols.add(col)

    # Identify free variables: columns without pivots.
    free_vars_indices = []
    for c in range(N):
        if c not in basic_vars_cols:
            free_vars_indices.append(c)
    
    num_free_vars = len(free_vars_indices)
    basis = [] # Stores basis vectors of the null space as integers.

    # Construct basis vectors for the null space. Each basis vector corresponds to one free variable.
    for i in range(num_free_vars):
        current_free_var_idx = free_vars_indices[i]
        
        # Start constructing the basis vector. Set the bit corresponding to the current free variable to 1.
        b_val = (1 << current_free_var_idx) 
        
        # Determine values for basic variables based on the RREF equations.
        # Iterate through the pivot rows (0 to rank-1). Each row defines one basic variable.
        for r in range(rank): 
            pivot_col = pivot_cols[r] # Index of the basic variable defined by row `r`.
            
            # The equation from row `r` of RREF is: x_pivot_col + sum_{k in FreeVars} A_rref[r][k] * x_k = 0
            # In GF(2), this means x_pivot_col = sum_{k in FreeVars} A_rref[r][k] * x_k
            # For the basis vector associated with `current_free_var_idx`, we set x_current_free_var_idx = 1
            # and all other free variables x_k = 0 (for k != current_free_var_idx).
            # Thus, x_pivot_col = A_rref[r][current_free_var_idx].
            
            # Check the coefficient A_rref[r][current_free_var_idx].
            if (A_rref[r] >> current_free_var_idx) & 1:
                 # If the coefficient is 1, then the value of the basic variable x_pivot_col is 1.
                 # Set the corresponding bit in the basis vector representation.
                 b_val |= (1 << pivot_col)
        
        basis.append(b_val) # Add the constructed basis vector to the list.

    # Check if a valid assignment exists and output the result.
    if num_free_vars == 0:
         # If dimension of null space is 0, the only solution to Ax=0 is the zero vector (x=0).
         # This means we must assign X_v = 0 for all v.
         # This is only allowed if N=0. But problem constraints say N>=1.
         # An assignment of all zeros is invalid because X_v must be >= 1.
         # However, if the graph originally had no edges (M=0), then the constraints 
         # are vacuously satisfied for any assignment. In this case, we can assign X_v = 1 for all v.
         if not is_any_edge and N > 0: # Graph had no edges, N>0. M=0 case.
              print("Yes")
              print(*( [1] * N )) # Assign 1 to all vertices works.
         # elif N == 0: # Trivial case N=0 is technically valid but excluded by constraints
         #      print("Yes")
         else: # Graph has edges, but the only solution is X=0. Cannot satisfy X_v >= 1.
              print("No")
    else: # num_free_vars > 0. There are non-zero solutions.
        possible = True
        # We need to ensure that for every vertex v, we can make X_v >= 1.
        # This is possible if and only if for every vertex v, there exists at least one basis vector
        # `b` in the computed basis such that the v-th component of `b` is 1.
        # If a vertex v has 0 in the v-th component of all basis vectors, then X_v = 0 regardless
        # of the choice of coefficients C_i, which is not allowed.
        for v in range(N):
            has_support = False # Flag indicates if v-th component is 1 for any basis vector
            for b_val in basis:
                 if (b_val >> v) & 1:
                     has_support = True
                     break
            if not has_support:
                 # Vertex v is 0 in all basis vectors. Implies X_v will always be 0.
                 possible = False
                 break
        
        if possible:
            print("Yes")
            # Construct one valid assignment using coefficients C_i = 2^i.
            X = [0] * N
            for v in range(N):
                val = 0
                # Calculate X_v = XOR sum_{i=1..d} (basis[i]_v * C_i)
                for i in range(num_free_vars):
                    C_i = 1 << i # Use 2^i for coefficients
                    # Check the v-th component (bit) of the i-th basis vector.
                    if (basis[i] >> v) & 1:
                         # If the v-th component is 1, XOR with the coefficient C_i.
                         val ^= C_i
                X[v] = val
            print(*(X)) # Print the assigned values for all vertices.
        else:
            # Cannot ensure X_v >= 1 for all v.
            print("No")

solve()