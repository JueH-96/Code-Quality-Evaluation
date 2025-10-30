def main():
    import sys

    data = sys.stdin.read().split()
    H = int(data[0])
    W = int(data[1])
    N = int(data[2])

    # Initialize all cells to white (False = white, True = black)
    grid = [[False]*W for _ in range(H)]

    # Directions: 0=up, 1=right, 2=down, 3=left
    # Each entry is (delta_row, delta_col)
    directions = [(-1,0), (0,1), (1,0), (0,-1)]
    
    # Starting position (0-based)
    row, col = 0, 0
    # Starting direction: up
    d = 0

    for _ in range(N):
        if not grid[row][col]:  # white cell
            grid[row][col] = True  # paint it black
            d = (d + 1) % 4  # rotate 90 deg clockwise
        else:  # black cell
            grid[row][col] = False  # paint it white
            d = (d - 1) % 4  # rotate 90 deg counterclockwise

        # move forward 1 cell
        dr, dc = directions[d]
        row = (row + dr) % H
        col = (col + dc) % W

    # Output the final grid
    for r in range(H):
        line = ""
        for c in range(W):
            if grid[r][c]:
                line += "#"
            else:
                line += "."
        print(line)

# Don't forget to call main!
if __name__ == "__main__":
    main()