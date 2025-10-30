def gauss_elimination_gf2(matrix, ncols):
    rows = len(matrix)
    
    r = 0  # Current row
    pivot_cols = []
    
    for c in range(ncols):
        # Find a row with a non-zero entry in column c
        for i in range(r, rows):
            if matrix[i][c] == 1:
                matrix[r], matrix[i] = matrix[i], matrix[r]
                break
        else:
            continue  # If no non-zero entry, move to the next column
        
        pivot_cols.append(c)
        
        # Eliminate in other rows
        for i in range(rows):
            if i != r and matrix[i][c] == 1:
                for j in range(c, ncols):
                    matrix[i][j] = (matrix[i][j] ^ matrix[r][j])  # XOR operation
        
        r += 1
        if r == rows:
            break
    
    return r, pivot_cols  # Return the rank of the matrix and pivot columns

def solve(N, M, edges):
    # Create the adjacency matrix
    A = [[0 for _ in range(N)] for _ in range(N)]
    for u, v in edges:
        u -= 1  # Convert to 0-indexed
        v -= 1
        A[u][v] = 1
        A[v][u] = 1
    
    # Compute the degree of each vertex
    degrees = [sum(A[i]) for i in range(N)]
    
    # Check if there's a vertex with degree 1
    if 1 in degrees:
        return "No", []
    
    # If all vertices have even degrees or have no edges, return a fixed value
    if all(d % 2 == 0 or d == 0 for d in degrees):
        return "Yes", [4] * N
    
    # Extract the rows of the constraint matrix corresponding to vertices with edges
    constraint_matrix = []
    vertices_with_edges = []
    for i in range(N):
        if degrees[i] > 0:
            constraint_matrix.append(A[i][:])
            vertices_with_edges.append(i)
    
    if not vertices_with_edges:
        return "Yes", [1] * N  # If no vertices have edges, any assignment is valid
    
    # Perform Gaussian elimination to check for a non-trivial solution
    rank, pivot_cols = gauss_elimination_gf2(constraint_matrix, N)
    
    # Check if the system is consistent
    for i in range(rank, len(constraint_matrix)):
        if any(constraint_matrix[i][j] == 1 for j in range(N)):
            return "No", []
    
    # If rank is less than the number of columns, there's a non-trivial solution
    if rank < N:
        # Construct a valid assignment
        solution = [0] * N
        free_cols = [j for j in range(N) if j not in pivot_cols]
        
        # Set the free variables to ensure non-triviality
        for j in free_cols:
            solution[j] = 4
        
        # Compute the values of the pivot variables
        for i in range(rank-1, -1, -1):
            pivot_col = pivot_cols[i]
            value = 0
            for j in range(pivot_col+1, N):
                if constraint_matrix[i][j] == 1:
                    value ^= solution[j]
            solution[pivot_col] = value
        
        # Ensure all values are non-zero (between 1 and 2^60 - 1)
        for i in range(N):
            if solution[i] == 0:
                solution[i] = 4
        
        return "Yes", solution
    
    return "No", []  # If no non-trivial solution exists

# Read inputs
N, M = map(int, input().split())
edges = []
for _ in range(M):
    u, v = map(int, input().split())
    edges.append((u, v))

answer, values = solve(N, M, edges)
print(answer)
if answer == "Yes":
    print(" ".join(map(str, values)))