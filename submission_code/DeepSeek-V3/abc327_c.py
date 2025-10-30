# YOUR CODE HERE
def is_valid_sudoku(grid):
    # Check rows
    for row in grid:
        if sorted(row) != list(range(1, 10)):
            return False
    # Check columns
    for col in range(9):
        column = [grid[row][col] for row in range(9)]
        if sorted(column) != list(range(1, 10)):
            return False
    # Check 3x3 subgrids
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            subgrid = []
            for x in range(i, i+3):
                for y in range(j, j+3):
                    subgrid.append(grid[x][y])
            if sorted(subgrid) != list(range(1, 10)):
                return False
    return True

# Read the grid
grid = []
for _ in range(9):
    row = list(map(int, input().split()))
    grid.append(row)

# Check if the grid is a valid Sudoku
if is_valid_sudoku(grid):
    print("Yes")
else:
    print("No")