H, W, N = map(int, input().split())

# Initialize grid: False represents white, True represents black
grid = [[False for _ in range(W)] for _ in range(H)]
x, y = 0, 0  # Starting position (0-based)
direction = 0  # 0: up, 1: right, 2: down, 3: left

# Direction deltas: (dx, dy)
deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]

for _ in range(N):
    current = grid[x][y]
    if current:
        grid[x][y] = False
        direction = (direction - 1) % 4
    else:
        grid[x][y] = True
        direction = (direction + 1) % 4
    dx, dy = deltas[direction]
    x = (x + dx) % H
    y = (y + dy) % W

# Convert the grid to the required output format
for row in grid:
    print(''.join('#' if cell else '.' for cell in row))