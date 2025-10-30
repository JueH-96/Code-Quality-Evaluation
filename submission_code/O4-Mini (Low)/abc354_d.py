def main():
    import sys
    A,B,C,D = map(int, sys.stdin.readline().split())

    # Precompute T(rx, ry) = twice the black area in [0, rx] x [0, ry],
    # for rx, ry in {0,1,2}.
    # We do it once by brute subdivision of the small 2×2 cell.
    # The walls are x = n (integer), y = 2k (even), x+y = 2k.
    # We test the color at the midpoint of each region after subdividing.
    # Then we sum up the areas.
    T = [[0]*3 for _ in range(3)]
    # We'll subdivide the [0,2]×[0,2] cell by the lines x=1, y=0,2, x+y=2.
    # There are 5 regions total in the tile; we can tag them by a small
    # triangulation/rectangle mesh.  To keep code short we simply sample
    # a fine grid over [0,2]×[0,2] with very fine steps, decide color
    # per little cell, and add up area.  Since the tile is only 2×2, even
    # a 200×200 grid is fast and accurate to within 1e-6, and after we
    # multiply by 2 and round we get the exact integer.
    #
    # Note: we only need rx,ry up to 2, so we can do a single pass
    # for the full 2×2 and accumulate partial sums for the nine rectangles.
    import math

    # A function giving +1 if black, -1 if white, at (x,y).
    # A point is black iff the parity of
    #    floor(x) + floor(y/2) + floor((x+y)/2)
    # is even (origin‐region (0.5,0.5) is black).
    def color_sign(x,y):
        s = math.floor(x) + math.floor(y/2.0) + math.floor((x+y)/2.0)
        return 1 if (s%2)==0 else -1

    # We'll sample a Nx×Ny grid of tiny rectangles in [0,2]×[0,2].
    Nx = Ny = 400
    dx = 2.0 / Nx
    dy = 2.0 / Ny

    # Accumulate partial sums for all subrectangles with rx,ry in {0,1,2}
    # T[rx][ry] = sum_{i<x_index(=rx/dx), j<y_index(=ry/dy)} sign * dx*dy, then *2 and round
    accum = [[0.0]*3 for _ in range(3)]
    for i in range(Nx):
        cx = (i+0.5)*dx
        ix = 0 if cx<1 else 1 if cx<2 else 2
        for j in range(Ny):
            cy = (j+0.5)*dy
            jy = 0 if cy<1 else 1 if cy<2 else 2
            s = color_sign(cx, cy)
            area = dx*dy
            # add this little area*s to all accum[rx][ry] with rx>ix and ry>jy
            # i.e. for rx in {ix+1,2}, ry in {jy+1,2}
            for rx in range(ix+1, 3):
                for ry in range(jy+1, 3):
                    accum[rx][ry] += s * area

    # Now fill T by rounding 2*accum[rx][ry]
    for rx in range(3):
        for ry in range(3):
            T[rx][ry] = int(round(2 * accum[rx][ry]))

    # T[2][2] is the twice‐black‐area of a full 2×2 tile.
    TT = T[2][2]

    def G(X, Y):
        # returns twice black area in [0,X]×[0,Y]
        if X<=0 or Y<=0:
            return 0
        x2, rx = divmod(X, 2)
        y2, ry = divmod(Y, 2)
        res = x2*y2*TT
        res += x2*T[2][ry]
        res += y2*T[rx][2]
        res += T[rx][ry]
        return res

    ans = G(C, D) - G(A, D) - G(C, B) + G(A, B)
    print(ans)

if __name__ == "__main__":
    main()