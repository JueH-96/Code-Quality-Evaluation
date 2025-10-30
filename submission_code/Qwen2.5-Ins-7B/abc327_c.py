# YOUR CODE HERE
def check_grid(grid):
    for i in range(9):
        row = [grid[i][j] for j in range(9)]
        col = [grid[j][i] for j in range(9)]
        if len(set(row)) != 9 or len(set(col)) != 9:
            return False
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            subgrid = [grid[x][y] for x in range(i, i+3) for y in range(j, j+3)]
            if len(set(subgrid)) != 9:
                return False
    return True

grid = [input().split() for _ in range(9)]
if check_grid(grid):
    print("Yes")
else:
    print("No")