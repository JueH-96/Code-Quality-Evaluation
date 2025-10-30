def main():
    import sys
    data = sys.stdin.read().strip().split()
    H, W, N = map(int, data)

    # 0 = white, 1 = black
    grid = [[0] * W for _ in range(H)]

    # Directions: 0=up, 1=right, 2=down, 3=left
    dir = 0
    i, j = 0, 0

    for _ in range(N):
        if grid[i][j] == 0:
            # white: paint black, turn clockwise
            grid[i][j] = 1
            dir = (dir + 1) % 4
        else:
            # black: paint white, turn counterclockwise
            grid[i][j] = 0
            dir = (dir + 3) % 4

        # move forward
        if dir == 0:       # up
            i -= 1
            if i < 0:
                i = H - 1
        elif dir == 1:     # right
            j += 1
            if j >= W:
                j = 0
        elif dir == 2:     # down
            i += 1
            if i >= H:
                i = 0
        else:              # left
            j -= 1
            if j < 0:
                j = W - 1

    # output
    out = []
    for row in grid:
        line = ''.join('#' if cell == 1 else '.' for cell in row)
        out.append(line)
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()