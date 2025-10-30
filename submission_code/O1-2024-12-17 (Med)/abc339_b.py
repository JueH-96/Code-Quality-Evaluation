def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    H, W, N = map(int, input_data)

    # 0-based indexing for our grid
    grid = [[False]*W for _ in range(H)]  # False = white, True = black

    # Directions: 0=up, 1=right, 2=down, 3=left
    drow = [-1, 0, 1, 0]
    dcol = [0, 1, 0, -1]
    direction = 0
    r, c = 0, 0  # current position (0-based)

    for _ in range(N):
        if not grid[r][c]:  # white
            grid[r][c] = True  # paint black
            direction = (direction + 1) % 4  # rotate clockwise
        else:  # black
            grid[r][c] = False  # paint white
            direction = (direction + 3) % 4  # rotate counterclockwise

        # move forward
        r = (r + drow[direction]) % H
        c = (c + dcol[direction]) % W

    # Output the final grid
    for row in grid:
        print("".join('#' if cell else '.' for cell in row))

# Do not forget to call main()
if __name__ == "__main__":
    main()