# YOUR CODE HERE
def solve():
    N = int(input())
    P = list(map(int, input().split()))
    Q = list(map(int, input().split()))
    
    # Convert to 0-indexed
    P = [p - 1 for p in P]
    Q = [q - 1 for q in Q]
    
    # Create the grid
    grid = [[0] * N for _ in range(N)]
    
    # Create inverse permutations to know the order
    P_inv = [0] * N
    Q_inv = [0] * N
    for i in range(N):
        P_inv[P[i]] = i
        Q_inv[Q[i]] = i
    
    # Fill the grid
    # We'll use a strategy where we ensure that rows and columns maintain the required order
    # by setting bits to 1 when necessary to maintain lexicographic order
    
    # For each position (i, j), we need to consider:
    # - The row order: row P[k] should be less than row P[k+1]
    # - The column order: column Q[k] should be less than column Q[k+1]
    
    for i in range(N):
        for j in range(N):
            # Check if we need to set this position to 1 to maintain order
            row_order = P_inv[i]  # Order of row i in the final sorting
            col_order = Q_inv[j]  # Order of column j in the final sorting
            
            # We can use a simple heuristic: set to 1 if the sum of orders is large enough
            # This helps ensure proper separation between rows and columns
            if row_order + col_order >= N - 1:
                grid[i][j] = 1
    
    # Verify and adjust if needed
    # Check row constraints
    for k in range(N - 1):
        row1 = P[k]
        row2 = P[k + 1]
        # Ensure row1 < row2 lexicographically
        equal = True
        for j in range(N):
            if grid[row1][j] < grid[row2][j]:
                equal = False
                break
            elif grid[row1][j] > grid[row2][j]:
                # Fix: make row2 have a 1 where row1 has 0
                for jj in range(j + 1, N):
                    if grid[row1][jj] == 0:
                        grid[row2][jj] = 1
                        break
                equal = False
                break
        if equal:
            # If rows are equal, make row2 larger by adding a 1
            for j in range(N):
                if grid[row2][j] == 0:
                    grid[row2][j] = 1
                    break
    
    # Check column constraints
    for k in range(N - 1):
        col1 = Q[k]
        col2 = Q[k + 1]
        # Ensure col1 < col2 lexicographically
        equal = True
        for i in range(N):
            if grid[i][col1] < grid[i][col2]:
                equal = False
                break
            elif grid[i][col1] > grid[i][col2]:
                # Fix: make col2 have a 1 where col1 has 0
                for ii in range(i + 1, N):
                    if grid[ii][col1] == 0:
                        grid[ii][col2] = 1
                        break
                equal = False
                break
        if equal:
            # If columns are equal, make col2 larger by adding a 1
            for i in range(N):
                if grid[i][col2] == 0:
                    grid[i][col2] = 1
                    break
    
    # Print the grid
    for i in range(N):
        print(''.join(map(str, grid[i])))

solve()