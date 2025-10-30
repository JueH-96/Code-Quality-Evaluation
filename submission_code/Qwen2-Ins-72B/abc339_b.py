H, W, N = map(int, input().split())
d = [[0] * W for _ in range(H)]
x, y = 0, 0
dx, dy = 0, -1
for _ in range(N):
    if d[x][y] == 0:
        d[x][y] = 1
        dx, dy = dy, -dx
    else:
        d[x][y] = 0
        dx, dy = -dy, dx
    x = (x + dx) % H
    y = (y + dy) % W

for i in range(H):
    for j in range(W):
        if d[i][j] == 0:
            print('.', end='')
        else:
            print('#', end='')
    print()