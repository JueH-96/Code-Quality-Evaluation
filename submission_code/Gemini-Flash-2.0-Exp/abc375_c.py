def solve():
    n = int(input())
    grid = []
    for _ in range(n):
        grid.append(list(input()))

    for i in range(n // 2):
        for x in range(i, n - i):
            for y in range(i, n - i):
                grid[y][n - 1 - x] = grid[x][y]

    for row in grid:
        print("".join(row))

solve()