# YOUR CODE HERE
def simulate_toroidal_grid(H, W, N):
    # Initialize the grid with all white cells
    grid = [['.' for _ in range(W)] for _ in range(H)]
    
    # Directions: 0 = up, 1 = right, 2 = down, 3 = left
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    direction = 0  # Start facing upwards
    
    # Starting position
    x, y = 0, 0
    
    for _ in range(N):
        if grid[x][y] == '.':
            # Repaint black, rotate 90 degrees clockwise
            grid[x][y] = '#'
            direction = (direction + 1) % 4
        else:
            # Repaint white, rotate 90 degrees counterclockwise
            grid[x][y] = '.'
            direction = (direction - 1) % 4
        
        # Move forward in the current direction
        dx, dy = directions[direction]
        x = (x + dx) % H
        y = (y + dy) % W
    
    # Print the final grid
    for row in grid:
        print(''.join(row))

# Read input
import sys
input = sys.stdin.read
H, W, N = map(int, input().strip().split())

# Simulate and print the result
simulate_toroidal_grid(H, W, N)