# YOUR CODE HERE
H, W, N = map(int, input().split())

# Initialize grid
grid = [[0] * W for _ in range(H)]

# Takahashi's initial position and direction
x, y = 0, 0
dx, dy = -1, 0  # Initially facing upwards

for _ in range(N):
    # Change color of current cell
    grid[x][y] = 1 - grid[x][y]
    
    if grid[x][y] == 1:  # If painted black
        # Rotate 90° clockwise
        dx, dy = dy, -dx
    else:  # If painted white
        # Rotate 90° counterclockwise
        dx, dy = -dy, dx
    
    # Move forward (with toroidal wrapping)
    x = (x + dx) % H
    y = (y + dy) % W

# Print the final grid
for row in grid:
    print(''.join('#' if cell else '.' for cell in row))