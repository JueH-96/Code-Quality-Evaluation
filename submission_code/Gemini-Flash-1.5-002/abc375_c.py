n = int(input())
grid = [list(input()) for _ in range(n)]
new_grid = [['.' for _ in range(n)] for _ in range(n)]

for i in range(1, n // 2 + 1):
    temp_grid = [['.' for _ in range(n)] for _ in range(n)]
    for x in range(i - 1, n - i):
        for y in range(i - 1, n - i):
            temp_grid[x][n - 1 - y] = grid[y][x]
    grid = temp_grid

for row in grid:
    print("".join(row))