# YOUR CODE HERE
def rotate_outer(grid):
    n = len(grid)
    if n <= 1:
        return grid

    # Extract the outer elements
    top = grid[0]
    bottom = grid[-1][::-1]
    left = [grid[i][0] for i in range(1, n-1)]
    right = [grid[i][-1] for i in range(n-2, 0, -1)]

    # Combine all outer elements
    outer = right + top + left + bottom

    # Rotate the outer elements
    outer = outer[-1:] + outer[:-1]

    # Put the rotated elements back
    new_grid = [row[:] for row in grid]
    
    # Top and bottom
    new_grid[0] = outer[n-1:2*n-1]
    new_grid[-1] = outer[3*n-3:2*n-3:-1]
    
    # Left and right
    for i in range(1, n-1):
        new_grid[i][0] = outer[2*n-2+i]
        new_grid[i][-1] = outer[n-1-i]

    return new_grid

# Read input
n = int(input())
grid = [list(input().strip()) for _ in range(n)]

# Rotate the outer elements
rotated_grid = rotate_outer(grid)

# Print the result
for row in rotated_grid:
    print(''.join(row))