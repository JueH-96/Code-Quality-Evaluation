import sys

H, W, N = map(int, input().split())

# Initialize the grid with all white cells
grid = [['.' for _ in range(W)] for _ in range(H)]

# Initial position and direction
x, y = 0, 0
directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left
dir_idx = 0

for _ in range(N):
    if grid[x][y] == '.':
        grid[x][y] = '#'
        dir_idx = (dir_idx + 1) % 4  # Rotate 90 degrees clockwise
    else:
        grid[x][y] = '.'
        dir_idx = (dir_idx - 1) % 4  # Rotate 90 degrees counterclockwise

    dx, dy = directions[dir_idx]
    x = (x + dx) % H
    y = (y + dy) % W

# Print the final grid
for row in grid:
    print(''.join(row))