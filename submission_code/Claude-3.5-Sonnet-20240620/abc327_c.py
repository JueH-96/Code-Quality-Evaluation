# YOUR CODE HERE
def is_valid_sudoku(grid):
    # Check rows
    for row in grid:
        if set(row) != set(range(1, 10)):
            return False
    
    # Check columns
    for col in zip(*grid):
        if set(col) != set(range(1, 10)):
            return False
    
    # Check 3x3 sub-grids
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            sub_grid = [grid[x][y] for x in range(i, i+3) for y in range(j, j+3)]
            if set(sub_grid) != set(range(1, 10)):
                return False
    
    return True

# Read input
grid = []
for _ in range(9):
    row = list(map(int, input().split()))
    grid.append(row)

# Check if the grid is valid and print result
print("Yes" if is_valid_sudoku(grid) else "No")