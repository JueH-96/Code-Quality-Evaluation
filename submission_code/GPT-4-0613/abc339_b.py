H, W, N = map(int, input().split())
grid = [['.' for _ in range(W)] for _ in range(H)]
x, y = 0, 0
dx, dy = -1, 0
for _ in range(N):
    if grid[x][y] == '.':
        grid[x][y] = '#'
        dx, dy = dy, -dx
    else:
        grid[x][y] = '.'
        dx, dy = -dy, dx
    x = (x + dx) % H
    y = (y + dy) % W
for row in grid:
    print(''.join(row))