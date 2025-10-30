import sys
input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
N = int(data[2])

grid = [['.' for _ in range(W)] for _ in range(H)]

directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
direction_index = 0

x, y = 0, 0

for _ in range(N):
    if grid[x][y] == '.':
        grid[x][y] = '#'
        direction_index = (direction_index + 1) % 4
    else:
        grid[x][y] = '.'
        direction_index = (direction_index - 1) % 4

    dx, dy = directions[direction_index]
    nx, ny = x + dx, y + dy

    if nx < 0:
        nx = H - 1
    elif nx >= H:
        nx = 0
    if ny < 0:
        ny = W - 1
    elif ny >= W:
        ny = 0

    x, y = nx, ny

for row in grid:
    print(''.join(row))