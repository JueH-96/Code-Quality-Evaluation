def shift_outer_clockwise(grid, N):
    # Extract the outer layer
    top = grid[0]
    right = [row[-1] for row in grid[1:-1]]
    bottom = grid[-1][::-1]
    left = [row[0] for row in grid[-2:0:-1]]
    
    # Shift the outer layer clockwise
    top, right, bottom, left = left[0] + top[:-1], top[-1] + right[:-1], right[-1] + bottom[:-1], bottom[-1] + left[:-1]
    
    # Place the shifted outer layer back into the grid
    grid[0] = top
    for i in range(1, N - 1):
        grid[i][-1] = right[i - 1]
    grid[-1] = bottom[::-1]
    for i in range(N - 2, 0, -1):
        grid[i][0] = left[N - 2 - i]
    
    return grid

# Read input
N = int(input().strip())
grid = [list(input().strip()) for _ in range(N)]

# Shift the outer squares clockwise by one square each
shifted_grid = shift_outer_clockwise(grid, N)

# Print the resulting grid
for row in shifted_grid:
    print(''.join(row))