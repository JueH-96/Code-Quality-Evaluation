def solve():
    n = int(input())
    grid = [list(input()) for _ in range(n)]

    for i in range(n // 2):
        new_grid = [row[:] for row in grid]
        for x in range(i, n - i):
            for y in range(i, n - i):
                new_grid[y][n - 1 - x] = grid[x][y]
        grid = new_grid

    for row in grid:
        print("".join(row))

solve()