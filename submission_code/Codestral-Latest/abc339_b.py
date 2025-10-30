# YOUR CODE HERE
import sys

# Read input
H, W, N = map(int, sys.stdin.read().split())

# Initialize the grid with all cells white
grid = [['.' for _ in range(W)] for _ in range(H)]

# Directions: Up, Right, Down, Left
directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# Starting position and direction
x, y = 0, 0
direction = 0

# Perform the operations
for _ in range(N):
    if grid[x][y] == '.':
        # Repaint to black
        grid[x][y] = '#'
        # Rotate 90 degrees clockwise
        direction = (direction + 1) % 4
    else:
        # Repaint to white
        grid[x][y] = '.'
        # Rotate 90 degrees counterclockwise
        direction = (direction - 1) % 4

    # Move forward in the current direction
    dx, dy = directions[direction]
    x = (x + dx) % H
    y = (y + dy) % W

# Print the final grid
for row in grid:
    print(''.join(row))