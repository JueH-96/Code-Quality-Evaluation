def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    H, W, N = map(int, input_data)
    
    # Initialize the grid (False for white, True for black)
    grid = [[False] * W for _ in range(H)]
    
    # Define directions: 0: up, 1: right, 2: down, 3: left.
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    current_direction = 0  # starting facing "up"
    
    # Starting position (0-indexed)
    r, c = 0, 0
    
    for _ in range(N):
        if not grid[r][c]:
            # Current cell is white: repaint to black, turn 90° clockwise.
            grid[r][c] = True
            current_direction = (current_direction + 1) % 4
        else:
            # Current cell is black: repaint to white, turn 90° counterclockwise.
            grid[r][c] = False
            current_direction = (current_direction - 1) % 4
        
        # Move one step forward in the current direction, wrap around using modulo.
        dr, dc = directions[current_direction]
        r = (r + dr) % H
        c = (c + dc) % W
    
    # Prepare the output with '.' for white and '#' for black.
    output = []
    for row in grid:
        output.append(''.join('#' if cell else '.' for cell in row))
    
    sys.stdout.write('
'.join(output))


if __name__ == "__main__":
    main()