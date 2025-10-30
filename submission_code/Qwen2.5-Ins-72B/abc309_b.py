# YOUR CODE HERE
def shift_outer_squares(grid):
    n = len(grid)
    if n == 2:
        return [[grid[1][0], grid[0][0]], [grid[0][1], grid[1][1]]]
    
    # Extract the outer layer
    top = grid[0][:-1]
    right = [row[-1] for row in grid[:-1]]
    bottom = grid[-1][1:][::-1]
    left = [row[0] for row in grid[1:]][::-1]
    
    # Combine the outer layer in clockwise order
    outer = top + right + bottom + left
    
    # Shift the outer layer by one position
    shifted_outer = [outer[-1]] + outer[:-1]
    
    # Reconstruct the grid with the shifted outer layer
    new_grid = [[0] * n for _ in range(n)]
    
    # Place the top row
    new_grid[0] = [shifted_outer.pop(0)] + grid[0][1:-1] + [shifted_outer.pop(0)]
    
    # Place the right column
    for i in range(1, n-1):
        new_grid[i][-1] = shifted_outer.pop(0)
    
    # Place the bottom row
    new_grid[-1] = [shifted_outer.pop(0)] + grid[-1][1:-1] + [shifted_outer.pop(0)]
    
    # Place the left column
    for i in range(n-2, 0, -1):
        new_grid[i][0] = shifted_outer.pop(0)
    
    return new_grid

# Read input
n = int(input())
grid = [list(map(int, input().strip())) for _ in range(n)]

# Shift the outer squares
new_grid = shift_outer_squares(grid)

# Print the new grid
for row in new_grid:
    print(''.join(map(str, row)))