def main():
    import sys
    input = sys.stdin.readline

    # Read input sheets
    H_A, W_A = map(int, input().split())
    A = [input().rstrip() for _ in range(H_A)]
    H_B, W_B = map(int, input().split())
    B = [input().rstrip() for _ in range(H_B)]
    H_X, W_X = map(int, input().split())
    X = [input().rstrip() for _ in range(H_X)]

    # Collect coordinates of black cells in each sheet
    A_cells = [(i, j) for i in range(H_A) for j in range(W_A) if A[i][j] == '#']
    B_cells = [(i, j) for i in range(H_B) for j in range(W_B) if B[i][j] == '#']
    X_cells = {(i, j) for i in range(H_X) for j in range(W_X) if X[i][j] == '#'}

    # We will try all relative translations of A and B w.r.t. the cut sheet origin.
    # Let da_y, da_x shift A's (0,0) into the cut's coordinates, similarly for B.
    # Valid shifts must map every '#' of A/B to within [0..H_X) x [0..W_X)
    # and onto '#' positions of X. Then the union of those mapped cells must
    # cover exactly X_cells.

    # Precompute shift ranges. A's row i maps to i + da_y in [0,H_X), so
    # -i <= da_y <= H_X - 1 - i for all i in [0, H_A). 
    # Overall da_y in [ -H_A+1 .. H_X-1 ]. Similar for da_x, and B.
    min_da_y = -H_A + 1
    max_da_y = H_X - 1
    min_da_x = -W_A + 1
    max_da_x = W_X - 1

    min_db_y = -H_B + 1
    max_db_y = H_X - 1
    min_db_x = -W_B + 1
    max_db_x = W_X - 1

    # Try every shift combination
    for da_y in range(min_da_y, max_da_y + 1):
        for da_x in range(min_da_x, max_da_x + 1):
            # First check A maps legally onto X '#' cells
            mapped_A = set()
            okA = True
            for (ay, ax) in A_cells:
                y = ay + da_y
                x = ax + da_x
                if not (0 <= y < H_X and 0 <= x < W_X and (y, x) in X_cells):
                    okA = False
                    break
                mapped_A.add((y, x))
            if not okA:
                continue

            for db_y in range(min_db_y, max_db_y + 1):
                for db_x in range(min_db_x, max_db_x + 1):
                    # Check B likewise
                    mapped = set(mapped_A)
                    okB = True
                    for (by, bx) in B_cells:
                        y = by + db_y
                        x = bx + db_x
                        if not (0 <= y < H_X and 0 <= x < W_X and (y, x) in X_cells):
                            okB = False
                            break
                        mapped.add((y, x))
                    if not okB:
                        continue

                    # Finally, the union of mapped A and B must cover all X '#'
                    if mapped == X_cells:
                        print("Yes")
                        return

    # If no combination worked
    print("No")

if __name__ == "__main__":
    main()