def main():
    import sys

    # Read inputs
    H, W, N = map(int, sys.stdin.readline().split())
    
    # Initialize the grid: 0 for white, 1 for black
    grid = [[0]*W for _ in range(H)]
    
    # Directions: up (0), right (1), down (2), left (3)
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    # Starting position (0-based) and initial direction (up)
    r, c = 0, 0
    d = 0
    
    for _ in range(N):
        if grid[r][c] == 0:
            # Current cell is white => paint black, turn clockwise
            grid[r][c] = 1
            d = (d + 1) % 4
        else:
            # Current cell is black => paint white, turn counterclockwise
            grid[r][c] = 0
            d = (d - 1) % 4
        
        # Move forward one cell (toroidal wrap)
        dr, dc = directions[d]
        r = (r + dr) % H
        c = (c + dc) % W
    
    # Output the final grid
    for row in range(H):
        line = ""
        for col in range(W):
            line += "#" if grid[row][col] == 1 else "."
        print(line)

# Do not forget to call main().
main()