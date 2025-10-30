def move_takahashi(h, w, n):
    # Initialize the grid with all white cells
    grid = [['.' for _ in range(w)] for _ in range(h)]
    
    # Initial position and direction
    x, y = 0, 0
    dx, dy = -1, 0  # Facing upwards
    
    for _ in range(n):
        # Paint the current cell and change direction
        if grid[x][y] == '.':
            grid[x][y] = '#'
            dx, dy = dy, -dx  # Rotate 90 degrees clockwise
        else:
            grid[x][y] = '.'
            dx, dy = -dy, dx  # Rotate 90 degrees counterclockwise
        
        # Move forward one cell
        x, y = (x + dx) % h, (y + dy) % w
    
    return grid

# Read input
H, W, N = map(int, input().split())

# Get the final grid
final_grid = move_takahashi(H, W, N)

# Print the final grid
for row in final_grid:
    print(''.join(row))