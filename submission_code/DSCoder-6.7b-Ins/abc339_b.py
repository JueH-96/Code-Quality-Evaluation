H, W, N = map(int, input().split())
grid = [['.' for _ in range(W)] for _ in range(H)]
x, y, d = 0, 0, 0
dx, dy = [-1, 0, 1, 0], [0, 1, 0, -1]

for _ in range(N):
    if grid[x][y] == '.':
        grid[x][y] = '#'
        d = (d + 1) % 4
    else:
        grid[x][y] = '.'
        d = (d - 1) % 4
    x = (x + dx[d]) % H
    y = (y + dy[d]) % W

for row in grid:
    print(''.join(row))