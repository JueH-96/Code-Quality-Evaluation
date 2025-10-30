def main():
    import sys
    data = sys.stdin.read().strip().split()
    H = int(data[0])
    W = int(data[1])
    N = int(data[2])
    # 0 = white, 1 = black
    grid = [[0]*W for _ in range(H)]
    # Directions: 0=up,1=right,2=down,3=left
    # dr, dc for each direction
    dr = [-1, 0, 1, 0]
    dc = [0, 1, 0, -1]
    # Starting position (0-based)
    r = 0
    c = 0
    d = 0  # facing up
    for _ in range(N):
        if grid[r][c] == 0:
            # white: paint black, turn right
            grid[r][c] = 1
            d = (d + 1) & 3
        else:
            # black: paint white, turn left
            grid[r][c] = 0
            d = (d - 1) & 3
        # move forward
        r = (r + dr[d]) % H
        c = (c + dc[d]) % W
    # Output
    out = []
    for i in range(H):
        row = ''.join('#' if grid[i][j] else '.' for j in range(W))
        out.append(row)
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()