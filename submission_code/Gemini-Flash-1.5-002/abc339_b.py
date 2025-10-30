# YOUR CODE HERE
H, W, N = map(int, input().split())
grid = [['.' for _ in range(W)] for _ in range(H)]
r, c = 0, 0
direction = 0  # 0: up, 1: right, 2: down, 3: left
for _ in range(N):
    if grid[r][c] == '.':
        grid[r][c] = '#'
        direction = (direction + 1) % 4
    else:
        grid[r][c] = '.'
        direction = (direction - 1) % 4
    if direction == 0:
        r = (r - 1) % H
    elif direction == 1:
        c = (c + 1) % W
    elif direction == 2:
        r = (r + 1) % H
    else:
        c = (c - 1) % W
for row in grid:
    print(''.join(row))