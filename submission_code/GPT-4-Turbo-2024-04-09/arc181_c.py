def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    P = list(map(int, data[1:N+1]))
    Q = list(map(int, data[N+1:2*N+1]))
    
    # Adjust P and Q to be zero-indexed
    P = [p - 1 for p in P]
    Q = [q - 1 for q in Q]
    
    # Create the grid
    grid = [['0' * N for _ in range(N)]]
    
    # We will use a simple method to ensure the lexicographical order:
    # Fill the grid such that the i-th row in the order P has increasing number of '1's
    # and the j-th column in the order Q has increasing number of '1's
    
    # Initialize an empty grid
    grid = [['0'] * N for _ in range(N)]
    
    # Assign values to the grid
    for i in range(N):
        # Fill the P[i]-th row in increasing order of '1's
        for j in range(N):
            if j < i:
                grid[P[i]][j] = '1'
    
    # Now we need to ensure the columns are also in increasing order
    # We will transpose the grid and apply the same logic as rows but now for columns
    transposed_grid = list(zip(*grid))  # Transpose the grid to work on columns as rows
    
    # Create a new grid from the transposed grid to apply the column constraints
    new_transposed_grid = [['0'] * N for _ in range(N)]
    
    for j in range(N):
        # Fill the Q[j]-th column in increasing order of '1's
        for i in range(N):
            if i < j:
                new_transposed_grid[Q[j]][i] = '1'
    
    # Transpose back to get the final grid
    final_grid = list(zip(*new_transposed_grid))
    
    # Print the grid
    for row in final_grid:
        print(''.join(row))