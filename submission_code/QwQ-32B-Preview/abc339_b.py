# Read input
H, W, N = map(int, input().split())

# Initialize grid with all white
grid = [['.' for _ in range(W)] for _ in range(H)]

# Define directions: up, right, down, left
directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
# Initial position and direction
row, col = 0, 0
direction = 0  # starts facing up

for _ in range(N):
    if grid[row][col] == '.':
        # Repaint to black
        grid[row][col] = '#'
        # Rotate 90 degrees clockwise
        direction = (direction + 1) % 4
    else:
        # Repaint to white
        grid[row][col] = '.'
        # Rotate 90 degrees counterclockwise
        direction = (direction - 1) % 4
    # Move forward
    delta_row, delta_col = directions[direction]
    row = (row + delta_row) % H
    col = (col + delta_col) % W

# Print the grid
for r in range(H):
    print(''.join(grid[r]))