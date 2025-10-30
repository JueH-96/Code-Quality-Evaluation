import sys
import threading

def main():
    import sys
    data = sys.stdin
    readline = data.readline

    N, Q = map(int, readline().split())
    # Build 2D prefix sum of the base N x N tile
    ps = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(N):
        row = readline().strip()
        cum = 0
        for j, c in enumerate(row):
            # convert to 1 if black, 0 if white
            cum += (c == 'B')
            # ps[i+1][j+1] = ps[i][j+1] + sum of this row up to j
            ps[i+1][j+1] = ps[i][j+1] + cum

    Sfull = ps[N][N]  # total blacks in one full N x N tile

    out = []
    # local copies for speed
    ps_loc = ps
    S_loc = Sfull
    n_loc = N

    for _ in range(Q):
        A, B, C, D = map(int, readline().split())
        # we want sum on [A..C] x [B..D], inclusive
        # define F(x,y) = sum on [0..x-1] x [0..y-1]
        x1, y1 = A, B
        x2, y2 = C + 1, D + 1

        # compute F(x2,y2)
        tx = x2 // n_loc
        rx = x2 - tx * n_loc
        ty = y2 // n_loc
        ry = y2 - ty * n_loc
        f2 = tx * ty * S_loc + tx * ps_loc[n_loc][ry] + ty * ps_loc[rx][n_loc] + ps_loc[rx][ry]

        # compute F(x1,y2)
        tx = x1 // n_loc
        rx = x1 - tx * n_loc
        # ty, ry are same as above
        f1 = tx * ty * S_loc + tx * ps_loc[n_loc][ry] + ty * ps_loc[rx][n_loc] + ps_loc[rx][ry]

        # compute F(x2,y1)
        # tx, rx same as for x2
        ty = y1 // n_loc
        ry = y1 - ty * n_loc
        f3 = tx * ty * S_loc + tx * ps_loc[n_loc][ry] + ty * ps_loc[rx][n_loc] + ps_loc[rx][ry]

        # compute F(x1,y1)
        tx = x1 // n_loc
        rx = x1 - tx * n_loc
        # ty, ry same as for y1
        f4 = tx * ty * S_loc + tx * ps_loc[n_loc][ry] + ty * ps_loc[rx][n_loc] + ps_loc[rx][ry]

        # inclusion-exclusion
        ans = f2 - f1 - f3 + f4
        out.append(str(ans))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()