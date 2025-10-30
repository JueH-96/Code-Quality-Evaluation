def solve():
    n = int(input())
    grid = [[0] * n for _ in range(n)]
    center = (n + 1) // 2 - 1
    grid[center][center] = 'T'
    
    num = 1
    for i in range(center):
        for j in range(i, n - i):
            grid[i][j] = num
            num += 1
        for j in range(i + 1, n - i):
            grid[j][n - i - 1] = num
            num += 1
        for j in range(n - i - 2, i - 1, -1):
            grid[n - i - 1][j] = num
            num += 1
        for j in range(n - i - 2, i, -1):
            grid[j][i] = num
            num += 1

    for row in grid:
        print(*row)

solve()