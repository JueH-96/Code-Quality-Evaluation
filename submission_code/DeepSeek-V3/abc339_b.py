# YOUR CODE HERE
H, W, N = map(int, input().split())

# Initialize the grid
grid = [['.' for _ in range(W)] for _ in range(H)]

# Initial position and direction
x, y = 0, 0
direction = 0  # 0: up, 1: right, 2: down, 3: left

# Direction vectors for movement
dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]

for _ in range(N):
    if grid[x][y] == '.':
        # Paint black
        grid[x][y] = '#'
        # Rotate 90 degrees clockwise
        direction = (direction + 1) % 4
    else:
        # Paint white
        grid[x][y] = '.'
        # Rotate 90 degrees counterclockwise
        direction = (direction - 1) % 4
    # Move forward
    x = (x + dx[direction]) % H
    y = (y + dy[direction]) % W

# Print the grid
for row in grid:
    print(''.join(row))