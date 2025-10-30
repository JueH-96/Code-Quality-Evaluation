n = int(input())
grid = [list(input().strip()) for _ in range(n)]

for i in range(1, (n // 2) + 1):
    # Create a new grid for this step
    new_grid = [row.copy() for row in grid]
    # Iterate over the subgrid for this step (0-based)
    # The subgrid is from i-1 to (n - i) inclusive in 0-based
    for x in range(i-1, n - i + 1):
        for y in range(i-1, n - i + 1):
            # Compute the destination in 0-based
            dest_x = y
            dest_y = n - x - 1  # because N+1 - (x+1) in 1-based is n - x in 0-based
            # Update the new_grid
            new_grid[dest_x][dest_y] = grid[x][y]
    # Replace grid with new_grid
    grid = new_grid

for row in grid:
    print(''.join(row))