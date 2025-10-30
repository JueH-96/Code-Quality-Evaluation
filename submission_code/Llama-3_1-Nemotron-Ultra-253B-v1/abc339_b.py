H, W, N = map(int, input().split())

grid = [[False for _ in range(W)] for _ in range(H)]
pos_i, pos_j = 0, 0
direction = 0  # 0: up, 1: right, 2: down, 3: left

for _ in range(N):
    current_color = grid[pos_i][pos_j]
    grid[pos_i][pos_j] = not current_color

    if current_color:
        direction = (direction - 1) % 4
    else:
        direction = (direction + 1) % 4

    if direction == 0:
        di, dj = -1, 0
    elif direction == 1:
        di, dj = 0, 1
    elif direction == 2:
        di, dj = 1, 0
    else:
        di, dj = 0, -1

    pos_i = (pos_i + di) % H
    pos_j = (pos_j + dj) % W

for row in grid:
    print(''.join('#' if cell else '.' for cell in row))