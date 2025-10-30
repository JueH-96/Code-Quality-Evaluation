def main():
    import sys
    input = sys.stdin.readline

    # Read input for sheet A
    H_A, W_A = map(int, input().split())
    A = [input().rstrip('
') for _ in range(H_A)]
    # Read input for sheet B
    H_B, W_B = map(int, input().split())
    B = [input().rstrip('
') for _ in range(H_B)]
    # Read input for sheet X
    H_X, W_X = map(int, input().split())
    X = [input().rstrip('
') for _ in range(H_X)]

    # Collect black squares (i, j) for each sheet
    A_blk = {(i, j) for i in range(H_A) for j in range(W_A) if A[i][j] == '#'}
    B_blk = {(i, j) for i in range(H_B) for j in range(W_B) if B[i][j] == '#'}
    X_blk = {(i, j) for i in range(H_X) for j in range(W_X) if X[i][j] == '#'}

    # Quick helper to get the bounding rectangle of a set of (r, c) coordinates
    def bounding_rect(cells):
        rmin = min(r for r, _ in cells)
        rmax = max(r for r, _ in cells)
        cmin = min(c for _, c in cells)
        cmax = max(c for _, c in cells)
        return rmin, rmax, cmin, cmax

    # Number of black squares in X
    size_X = len(X_blk)

    # We will fix sheet A at (0,0) and try placing B at offsets (r, c)
    # Restrict the search to offsets in [-20..20], enough to ensure
    # that if the bounding rectangle exceeds H_X,W_X, it's invalid anyway.
    for r_off in range(-20, 21):
        for c_off in range(-20, 21):
            # Build the union of black squares for A and the translated B
            union_squares = set(A_blk)
            for (br, bc) in B_blk:
                union_squares.add((r_off + br, c_off + bc))

            # If the total number of black squares doesn't match X, skip
            if len(union_squares) != size_X:
                continue

            # Get bounding rectangle of union
            rmin, rmax, cmin, cmax = bounding_rect(union_squares)
            height = rmax - rmin + 1
            width = cmax - cmin + 1

            # If bounding rectangle is bigger than X can accommodate, skip
            if height > H_X or width > W_X:
                continue

            # Now, we need to see if there's a way to cut an H_X x W_X piece
            # that contains exactly these black squares in positions matching X.

            # The sub-sheet top-left corner (u, v) must satisfy:
            #   u <= rmin  and  u + H_X - 1 >= rmax  =>  rmax - (H_X - 1) <= u <= rmin
            #   v <= cmin  and  v + W_X - 1 >= cmax  =>  cmax - (W_X - 1) <= v <= cmin
            u_min = rmax - (H_X - 1)
            u_max = rmin
            v_min = cmax - (W_X - 1)
            v_max = cmin

            # If no valid range, skip
            if u_min > u_max or v_min > v_max:
                continue

            # Try all valid placements of the H_X x W_X cutout
            for u in range(u_min, u_max + 1):
                for v in range(v_min, v_max + 1):
                    # Check whether this cutout matches X exactly
                    # and includes all union_squares inside
                    # We'll map each black square (rr, cc) in union_squares
                    # to local coords (i, j) = (rr-u, cc-v) and check X[i][j].
                    matched = True
                    for (rr, cc) in union_squares:
                        i = rr - u
                        j = cc - v
                        # Must be within [0..H_X-1] x [0..W_X-1]
                        if not (0 <= i < H_X and 0 <= j < W_X):
                            matched = False
                            break
                        # Must match a black cell in X
                        if X[i][j] == '.':
                            matched = False
                            break

                    if not matched:
                        continue

                    # Because the cardinalities match (union_squares == size_X),
                    # if all union squares align with '#' in X, then
                    # we have a perfect match. We don't need an extra check
                    # for missing black squares because the counts match.
                    print("Yes")
                    return

    # If we exhaust all possibilities with no success:
    print("No")


# Do not forget to call main()
if __name__ == "__main__":
    main()