def paint_grid(H, W, N):
    # Initialize the grid with all white cells
    grid = [['.' for _ in range(W)] for _ in range(H)]
    
    # Directions: Up, Right, Down, Left
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    current_direction = 0  # Start facing Up
    x, y = 0, 0  # Start at (1, 1) which is (0, 0) in 0-indexed
    
    for _ in range(N):
        if grid[x][y] == '.':
            grid[x][y] = '#'  # Paint black
            current_direction = (current_direction + 1) % 4  # Rotate clockwise
        else:
            grid[x][y] = '.'  # Paint white
            current_direction = (current_direction - 1) % 4  # Rotate counterclockwise
        
        # Move in the current direction
        x = (x + directions[current_direction][0]) % H
        y = (y + directions[current_direction][1]) % W
    
    # Print the final grid
    for row in grid:
        print(''.join(row))

# Read input
import sys
input = sys.stdin.read
H, W, N = map(int, input().strip().split())
paint_grid(H, W, N)