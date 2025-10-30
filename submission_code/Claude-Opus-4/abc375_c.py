# YOUR CODE HERE
N = int(input())
grid = []
for _ in range(N):
    grid.append(list(input().strip()))

# Perform operations for i = 1, 2, ..., N/2
for i in range(1, N//2 + 1):
    # Create a temporary grid to store the new values
    new_grid = [row[:] for row in grid]
    
    # For all pairs (x, y) where i <= x, y <= N+1-i
    for x in range(i, N+2-i):
        for y in range(i, N+2-i):
            # Replace color at (y, N+1-x) with color at (x, y)
            # Note: we use 0-based indexing, so adjust accordingly
            src_row = x - 1
            src_col = y - 1
            dst_row = y - 1
            dst_col = (N + 1 - x) - 1
            
            new_grid[dst_row][dst_col] = grid[src_row][src_col]
    
    # Update the grid
    grid = new_grid

# Print the result
for row in grid:
    print(''.join(row))