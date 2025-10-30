# YOUR CODE HERE

def check_sudoku(grid):
    # Check rows
    for row in grid:
        if sorted(row) != list(range(1,10)):
            return "No"

    # Check columns
    for col in range(9):
        if sorted(grid[i][col] for i in range(9)) != list(range(1,10)):
            return "No"

    # Check 3x3 grids
    for i in range(0,9,3):
        for j in range(0,9,3):
            grid_part = [grid[x][y] for x in range(i,i+3) for y in range(j,j+3)]
            if sorted(grid_part) != list(range(1,10)):
                return "No"

    return "Yes"

grid = [list(map(int, input().split())) for _ in range(9)]
print(check_sudoku(grid))