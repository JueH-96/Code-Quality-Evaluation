H, W, N = map(int, input().split())
grid = [[False]*W for _ in range(H)]
x, y = 0, 0  # starting position (0-based index)
dir = 0  # Up=0, Right=1, Down=2, Left=3
dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]

for _ in range(N):
    if not grid[x][y]:
        grid[x][y] = True
        dir = (dir + 1) % 4
    else:
        grid[x][y] = False
        dir = (dir - 1) % 4
    x = (x + dx[dir]) % H
    y = (y + dy[dir]) % W

for i in range(H):
    row = ''
    for j in range(W):
        if grid[i][j]:
            row += '#'
        else:
            row += '.'
    print(row)