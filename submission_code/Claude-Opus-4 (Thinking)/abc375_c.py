N = int(input())
grid = []
for _ in range(N):
    grid.append(list(input().strip()))

for i in range(1, N//2 + 1):
    # Create a new grid for this iteration
    new_grid = [row[:] for row in grid]
    
    # For all x, y from i to N+1-i (1-indexed)
    for x in range(i, N + 2 - i):
        for y in range(i, N + 2 - i):
            # Replace (y, N+1-x) with (x, y)
            # Convert to 0-indexing: 
            # Source: (x, y) → grid[x-1][y-1]
            # Destination: (y, N+1-x) → grid[y-1][N-x]
            new_grid[y-1][N-x] = grid[x-1][y-1]
    
    # Update the grid
    grid = new_grid

# Print the result
for row in grid:
    print(''.join(row))