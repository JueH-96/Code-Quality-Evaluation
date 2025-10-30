# YOUR CODE HERE
def simulate_takahashi(H, W, N):
    # Initialize the grid with all cells white (False)
    grid = [[False] * W for _ in range(H)]
    
    # Directions: 0 = up, 1 = right, 2 = down, 3 = left
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    # Start position and direction
    x, y = 0, 0
    direction = 0
    
    for _ in range(N):
        # Check the current cell color
        if not grid[x][y]:  # White
            # Paint it black
            grid[x][y] = True
            # Rotate 90 degrees clockwise
            direction = (direction + 1) % 4
        else:  # Black
            # Paint it white
            grid[x][y] = False
            # Rotate 90 degrees counterclockwise
            direction = (direction - 1) % 4
        
        # Move forward in the current direction
        dx, dy = directions[direction]
        x = (x + dx) % H
        y = (y + dy) % W
    
    # Output the grid
    for row in grid:
        print(''.join('#' if cell else '.' for cell in row))

# Read input
import sys
input = sys.stdin.read
H, W, N = map(int, input().strip().split())

# Simulate and print the result
simulate_takahashi(H, W, N)