def main():
    import sys
    input = sys.stdin.readline

    H, W, N = map(int, input().split())
    # 0 = white, 1 = black
    grid = [[0]*W for _ in range(H)]
    # directions: 0=up,1=right,2=down,3=left
    dr = [-1, 0, 1, 0]
    dc = [0, 1, 0, -1]

    # starting position (0,0) in 0-based, facing up (d=0)
    r = 0
    c = 0
    d = 0

    for _ in range(N):
        if grid[r][c] == 0:
            # white: paint black, turn right
            grid[r][c] = 1
            d = (d + 1) % 4
        else:
            # black: paint white, turn left
            grid[r][c] = 0
            d = (d + 3) % 4
        # move forward on the torus
        r = (r + dr[d]) % H
        c = (c + dc[d]) % W

    # output
    out = []
    for i in range(H):
        row = []
        for j in range(W):
            row.append('#' if grid[i][j] else '.')
        out.append(''.join(row))
    sys.stdout.write('
'.join(out))

main()