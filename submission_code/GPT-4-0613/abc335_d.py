N = int(input())
grid = [['' for _ in range(N)] for _ in range(N)]
x, y = N // 2, N // 2
dx, dy = [0, 1, 0, -1], [1, 0, -1, 0]
d = 0
grid[x][y] = 'T'
for i in range(1, N * N):
    if (i - 1) % (N // 2) == 0 and i != 1:
        d = (d + 1) % 4
    x += dx[d]
    y += dy[d]
    grid[x][y] = str(i)
for row in grid:
    print(' '.join(row))