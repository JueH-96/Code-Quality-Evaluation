def main():
    import sys
    input = sys.stdin.readline
    
    # Read the grid dimensions and the number of operations
    H, W, N = map(int, input().split())
    
    # Initialize the grid with all white cells (False indicates white, True indicates black)
    grid = [[False] * W for _ in range(H)]
    
    # Define movement vectors for directions:
    # 0: up, 1: right, 2: down, 3: left
    dr = [-1, 0, 1, 0]
    dc = [0, 1, 0, -1]
    
    # Initial position (0-indexed): (0, 0) corresponding to (1, 1) in the problem.
    # Initial direction is 0 (upwards).
    r, c = 0, 0
    d = 0
    
    # Process the N operations
    for _ in range(N):
        if not grid[r][c]:  # current cell is white
            grid[r][c] = True  # repaint to black
            d = (d + 1) % 4    # rotate 90° clockwise
        else:               # current cell is black
            grid[r][c] = False  # repaint to white
            d = (d - 1) % 4     # rotate 90° counterclockwise
        
        # Move forward one cell in the current direction, using modulo for toroidal grid
        r = (r + dr[d]) % H
        c = (c + dc[d]) % W
    
    # Output the final grid state:
    # Use '#' for black cells and '.' for white cells.
    for row in grid:
        print("".join('#' if cell else '.' for cell in row))
                
if __name__ == '__main__':
    main()