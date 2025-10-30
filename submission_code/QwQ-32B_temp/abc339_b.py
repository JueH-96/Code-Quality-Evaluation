directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

H, W, N = map(int, input().split())

grid = [[False for _ in range(W)] for _ in range(H)]
row, col = 0, 0
direction = 0  # 0 is up

for _ in range(N):
    current_color = grid[row][col]
    # Flip the color
    grid[row][col] = not current_color

    # Rotate direction based on original color
    if not current_color:  # was white
        direction = (direction + 1) % 4
    else:  # was black
        direction = (direction - 1) % 4

    # Move in the new direction
    dr, dc = directions[direction]
    row = (row + dr) % H
    col = (col + dc) % W

# Print the grid
for r in grid:
    print(''.join('#' if cell else '.' for cell in r))