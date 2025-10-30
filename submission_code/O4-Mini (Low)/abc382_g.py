def main():
    import sys
    input = sys.stdin.readline

    T = int(input())
    out = []

    for _ in range(T):
        K, Sx, Sy, Tx, Ty = map(int, input().split())

        # Determine the (i,j,k) tile-coordinates for a point (x+0.5, y+0.5)
        def tile_coords(x, y):
            # which block in i,j
            i = x // K
            j = y // K
            # parity of (i+j)
            if ((i + j) & 1) == 0:
                # even parity: tile is horizontal strip, so y-offset gives k
                k = y - j*K
            else:
                # odd parity: tile is vertical strip, so x-offset gives k
                k = x - i*K
            return int(i), int(j), int(k)

        Si, Sj, Sk = tile_coords(Sx, Sy)
        Ti, Tj, Tk = tile_coords(Tx, Ty)

        # We need to go from (Si,Sj,Sk) to (Ti,Tj,Tk)
        # Moves allowed:
        #  - change k by ±1 (inside same cell)
        #  - if you're at k=0 or k=K-1 and the tile parity allows, you can step to the adjacent cell in i or j.
        #
        # Observations:
        # 1) Every time you move cell (i or j), parity flips (since i+j changes by ±1).
        # 2) Horizontal moves are only possible on odd-parity tiles at k=0 (west) or k=K-1 (east).
        # 3) Vertical moves are only possible on even-parity tiles at k=0 (south) or k=K-1 (north).
        #
        # We can break the journey into three phases:
        #  A) Inside start cell adjust Sk -> some boundary k_surf = 0 or K-1 to exit in
        #     the direction of the first cell-step.
        #  B) Make the rectilinear path of |Δi| + |Δj| cell-steps, alternating direction
        #     as needed, always arriving at a boundary k of the next cell so we can step again.
        #  C) Inside end cell adjust boundary -> Tk.
        #
        # It turns out the minimum extra needed in A+C beyond |Sk - Tk| is at most 1,
        # and depends only on whether you need to "flip" parity an extra time before matching Tk.
        #
        di = Ti - Si
        dj = Tj - Sj
        steps_cells = abs(di) + abs(dj)

        if steps_cells == 0:
            # same cell, only adjust k internally
            ans = abs(Sk - Tk)
        else:
            # We must make at least one cell-step:
            # * One of them (the first) determines whether we need to go to k=0 or K-1 to exit.
            # * After each cell-step, parity flips and we arrive at that same boundary k.
            #   So throughout the steps, our k stays at that boundary.
            # * At the end of the last cell-step, we arrive at the end-cell also at that boundary,
            #   then we must move from boundary to Tk inside the end-cell.
            #
            # So total = (Sk -> boundary) + steps_cells + (boundary -> Tk).
            # Try both possible exit boundaries b in {0, K-1} and pick min.
            b0 = 0
            b1 = K-1
            best = None
            for b in (b0, b1):
                cost = abs(Sk - b) + steps_cells + abs(Tk - b)
                if best is None or cost < best:
                    best = cost
            ans = best

        out.append(str(ans))

    print("
".join(out))


if __name__ == "__main__":
    main()