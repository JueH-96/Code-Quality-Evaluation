def check_sudoku(grid):
    # Check rows
    for row in grid:
        if sorted(row) != list(range(1, 10)):
            return False

    # Check columns
    for col in range(9):
        column = [grid[row][col] for row in range(9)]
        if sorted(column) != list(range(1, 10)):
            return False

    # Check 3x3 sub-grids
    for box_row in range(3):
        for box_col in range(3):
            sub_grid = [grid[row][col] for row in range(box_row * 3, box_row * 3 + 3)
                                        for col in range(box_col * 3, box_col * 3 + 3)]
            if sorted(sub_grid) != list(range(1, 10)):
                return False

    return True

# Read input
grid = []
for _ in range(9):
    row = list(map(int, input().split()))
    grid.append(row)

# Check if the grid is a valid Sudoku
if check_sudoku(grid):
    print("Yes")
else:
    print("No")