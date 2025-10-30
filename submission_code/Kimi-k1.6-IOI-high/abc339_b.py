H, W, N = map(int, input().split())

# Initialize grid with all white (False)
grid = [[False for _ in range(W)] for _ in range(H)]

r, c = 0, 0  # Starting position (0-based)
direction = 0  # 0: up, 1: right, 2: down, 3: left
directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

for _ in range(N):
    current = grid[r][c]
    # Toggle the color
    grid[r][c] = not current
    # Rotate direction
    if current:
        direction = (direction - 1) % 4
    else:
        direction = (direction + 1) % 4
    # Move forward
    dr, dc = directions[direction]
    r = (r + dr) % H
    c = (c + dc) % W

# Generate output
for row in grid:
    print(''.join('#' if cell else '.' for cell in row))