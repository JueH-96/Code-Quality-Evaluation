N = int(input())
grid = []
for _ in range(N):
    grid.append(list(input().strip()))

# Perform operations for each layer i from 1 to N/2
for i in range(1, N//2 + 1):
    # Create a copy to store the new values
    new_grid = [row[:] for row in grid]
    
    # For all pairs (x,y) where i ≤ x,y ≤ N+1-i
    for x in range(i, N+2-i):  # 1-based range
        for y in range(i, N+2-i):  # 1-based range
            # Replace cell (y, N+1-x) with color from cell (x,y)
            # Convert to 0-based indexing
            src_row = x - 1
            src_col = y - 1
            dst_row = y - 1
            dst_col = N + 1 - x - 1  # N - x
            
            new_grid[dst_row][dst_col] = grid[src_row][src_col]
    
    grid = new_grid

# Print the result
for row in grid:
    print(''.join(row))