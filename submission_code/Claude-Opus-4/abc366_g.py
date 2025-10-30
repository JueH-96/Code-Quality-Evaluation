def solve():
    N, M = map(int, input().split())
    
    # Build adjacency list
    adj = [[] for _ in range(N)]
    for _ in range(M):
        u, v = map(int, input().split())
        u -= 1  # 0-indexed
        v -= 1
        adj[u].append(v)
        adj[v].append(u)
    
    # Special case: no edges
    if M == 0:
        print("Yes")
        print(" ".join(["1"] * N))
        return
    
    # Check if graph has a solution
    # Key insight: For each connected component, we can assign values
    # such that the constraints are satisfied if and only if
    # the component is bipartite or has an even number of edges
    
    # Actually, let's use a different approach:
    # We'll use the fact that for any tree, we can always find a solution
    # For graphs with cycles, we need to check if the system is consistent
    
    # Build constraint matrix
    # For each vertex v with degree >= 1, we have constraint:
    # XOR of X[u] for all u adjacent to v = 0
    
    # We'll use Gaussian elimination over GF(2)
    # Matrix A where A[i][j] = 1 if vertex j is adjacent to vertex i
    
    constraints = []
    for v in range(N):
        if len(adj[v]) > 0:
            row = [0] * N
            for u in adj[v]:
                row[u] = 1
            constraints.append(row)
    
    # Gaussian elimination over GF(2)
    m = len(constraints)
    n = N
    
    # Make a copy for manipulation
    matrix = [row[:] for row in constraints]
    
    # Forward elimination
    pivot_row = 0
    pivot_cols = []
    
    for col in range(n):
        # Find pivot
        found = False
        for row in range(pivot_row, m):
            if matrix[row][col] == 1:
                # Swap rows
                matrix[pivot_row], matrix[row] = matrix[row], matrix[pivot_row]
                found = True
                break
        
        if not found:
            continue
        
        pivot_cols.append(col)
        
        # Eliminate
        for row in range(m):
            if row != pivot_row and matrix[row][col] == 1:
                for c in range(n):
                    matrix[row][c] ^= matrix[pivot_row][c]
        
        pivot_row += 1
    
    # Check for inconsistency (would mean no solution)
    # In this problem, the RHS is always 0, so we just need to check
    # if we have enough degrees of freedom
    
    # We have n - len(pivot_cols) free variables
    free_vars = []
    for i in range(n):
        if i not in pivot_cols:
            free_vars.append(i)
    
    # Assign values
    X = [0] * N
    
    # Set free variables to powers of 2 (to ensure linear independence)
    for i, var in enumerate(free_vars):
        X[var] = 1 << i
    
    # Back substitution to find dependent variables
    for i in range(len(pivot_cols) - 1, -1, -1):
        col = pivot_cols[i]
        val = 0
        for j in range(col + 1, n):
            if matrix[i][j] == 1:
                val ^= X[j]
        X[col] = val
    
    # Check if all values are non-zero
    if any(x == 0 for x in X):
        # Try a different assignment
        # Add 1 to all values
        for i in range(N):
            X[i] = X[i] if X[i] > 0 else 1
    
    # Verify solution
    valid = True
    for v in range(N):
        if len(adj[v]) > 0:
            xor_sum = 0
            for u in adj[v]:
                xor_sum ^= X[u]
            if xor_sum != 0:
                valid = False
                break
    
    if not valid:
        print("No")
    else:
        print("Yes")
        print(" ".join(map(str, X)))

solve()