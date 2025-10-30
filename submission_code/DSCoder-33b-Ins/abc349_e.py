# YOUR CODE HERE

def check_win(grid):
    for i in range(3):
        if grid[i][0] == grid[i][1] == grid[i][2] != 0:
            return grid[i][0]
        if grid[0][i] == grid[1][i] == grid[2][i] != 0:
            return grid[0][i]
    if grid[0][0] == grid[1][1] == grid[2][2] != 0:
        return grid[0][0]
    if grid[0][2] == grid[1][1] == grid[2][0] != 0:
        return grid[0][2]
    return 0

def solve():
    grid = []
    for _ in range(3):
        grid.append(list(map(int, input().split())))
    total = sum(sum(row) for row in grid)
    if total % 2 == 0:
        return "Aoki"
    else:
        return "Takahashi"

print(solve())