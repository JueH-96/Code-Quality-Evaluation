# YOUR CODE HERE

def check_sudoku(grid):
    # Check rows
    for row in grid:
        if sorted(row) != list(range(1, 10)):
            return False

    # Check columns
    for col in range(9):
        if sorted([grid[row][col] for row in range(9)]) != list(range(1, 10)):
            return False

    # Check 3x3 grids
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            grid3x3 = [grid[x][y] for x in range(i, i + 3) for y in range(j, j + 3)]
            if sorted(grid3x3) != list(range(1, 10)):
                return False

    return True

grid = []
for _ in range(9):
    row = list(map(int, input().split()))
    grid.append(row)

if check_sudoku(grid):
    print("Yes")
else:
    print("No")