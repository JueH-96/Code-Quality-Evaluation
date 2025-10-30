def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    edges = []
    index = 2
    for _ in range(M):
        u = int(data[index])
        v = int(data[index+1])
        edges.append((u, v))
        index += 2
    
    # Build adjacency list
    adj = [[] for _ in range(N+1)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check if the graph is a complete graph or has certain properties
    # For a complete graph, all vertices must have the same value
    # For other graphs, it's more complex
    
    # First, check if the graph is a complete graph
    is_complete = True
    for u in range(1, N+1):
        if len(adj[u]) != N-1:
            is_complete = False
            break
    if is_complete:
        # All vertices must have the same value
        # Any value between 1 and 2^60 - 1 is acceptable
        value = 1
        print("Yes")
        print(" ".join([str(value)] * N))
        return
    
    # Now, handle other cases
    # We need to assign values such that for each vertex, the XOR of its neighbors is 0
    # This is equivalent to saying that the sum of the values of the neighbors is 0 in the XOR sense
    # This is a system of linear equations over the field GF(2)
    
    # We can represent the problem as a system of equations where for each vertex, the sum of its neighbors is 0
    # We can represent this as a matrix and solve it
    
    # Create a matrix where each row represents a vertex, and each column represents a variable (the value of a vertex)
    # The matrix will have N rows and N columns
    # For each vertex u, the row will have 1s in the columns corresponding to its neighbors
    
    # Initialize the matrix
    matrix = [[0] * N for _ in range(N)]
    for u in range(1, N+1):
        for v in adj[u]:
            matrix[u-1][v-1] = 1
    
    # We need to solve the system matrix * X = 0, where X is the vector of values
    # This is equivalent to finding a vector X in the null space of the matrix
    
    # To find a non-trivial solution, we need the matrix to have a non-trivial null space
    # This happens if the matrix is not full rank
    
    # We can perform Gaussian elimination to find the rank of the matrix
    
    # Perform Gaussian elimination
    rank = 0
    for col in range(N):
        # Find the row with the first 1 in this column
        pivot = -1
        for row in range(rank, N):
            if matrix[row][col] == 1:
                pivot = row
                break
        if pivot == -1:
            continue
        # Swap rows
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        # Eliminate this column in other rows
        for row in range(N):
            if row != rank and matrix[row][col] == 1:
                for c in range(col, N):
                    matrix[row][c] ^= matrix[rank][c]
        rank += 1
    
    # If the rank is less than N, there are non-trivial solutions
    if rank < N:
        # Assign arbitrary values to the free variables
        # For simplicity, assign 0 to all free variables except one, which we assign 1
        # Then, compute the dependent variables
        X = [0] * N
        # Find the free variables
        free_vars = []
        for row in range(rank):
            for col in range(N):
                if matrix[row][col] == 1:
                    break
            else:
                continue
            # The variable corresponding to this column is dependent
            # The free variables are those not in the pivot columns
        # Assign 1 to one of the free variables
        # For simplicity, assign 1 to the first free variable
        # The free variables are those columns not in the pivot columns
        pivot_cols = set()
        for row in range(rank):
            for col in range(N):
                if matrix[row][col] == 1:
                    pivot_cols.add(col)
                    break
        free_cols = set(range(N)) - pivot_cols
        if not free_cols:
            # No free variables, only the trivial solution
            # Check if the trivial solution is acceptable
            # The trivial solution is all zeros, but the values must be between 1 and 2^60 - 1
            # So, no solution
            print("No")
            return
        # Assign 1 to the first free variable
        first_free = min(free_cols)
        X[first_free] = 1
        # Now, compute the dependent variables
        for row in range(rank-1, -1, -1):
            # Find the pivot column
            pivot_col = -1
            for col in range(N):
                if matrix[row][col] == 1:
                    pivot_col = col
                    break
            if pivot_col == -1:
                continue
            # Compute the value of the pivot variable
            sum_val = 0
            for col in range(pivot_col+1, N):
                sum_val ^= matrix[row][col] * X[col]
            X[pivot_col] = sum_val
        # Now, check if all values are between 1 and 2^60 - 1
        # Since we assigned 1 to one variable and the rest are computed, it's possible that some are zero
        # If any value is zero, we need to adjust
        if 0 in X:
            # Find the first zero and assign a non-zero value
            # For simplicity, assign 1
            for i in range(N):
                if X[i] == 0:
                    X[i] = 1
                    # Recompute the dependent variables
                    for row in range(rank-1, -1, -1):
                        pivot_col = -1
                        for col in range(N):
                            if matrix[row][col] == 1:
                                pivot_col = col
                                break
                        if pivot_col == -1:
                            continue
                        sum_val = 0
                        for col in range(pivot_col+1, N):
                            sum_val ^= matrix[row][col] * X[col]
                        X[pivot_col] = sum_val
                    break
        # Now, check again
        if 0 in X:
            print("No")
            return
        print("Yes")
        print(" ".join(map(str, X)))
        return
    else:
        # The matrix is full rank, only the trivial solution exists
        # The trivial solution is all zeros, but the values must be between 1 and 2^60 - 1
        # So, no solution
        print("No")
        return

if __name__ == "__main__":
    main()