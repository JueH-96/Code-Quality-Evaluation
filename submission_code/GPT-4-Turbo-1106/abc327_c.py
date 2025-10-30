def is_valid_sudoku(grid):
    # Check rows and columns
    for i in range(9):
        if len(set(grid[i])) != 9 or len(set([grid[j][i] for j in range(9)])) != 9:
            return False

    # Check 3x3 subgrids
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            subgrid = [grid[x][y] for x in range(i, i+3) for y in range(j, j+3)]
            if len(set(subgrid)) != 9:
                return False

    return True

# Read the input
grid = [list(map(int, input().split())) for _ in range(9)]

# Check if the grid is a valid sudoku and print the result
print("Yes" if is_valid_sudoku(grid) else "No")