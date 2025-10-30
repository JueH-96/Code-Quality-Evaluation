def main():
    import sys

    # Read input
    H_A, W_A = map(int, sys.stdin.readline().split())
    A = [sys.stdin.readline().rstrip() for _ in range(H_A)]
    H_B, W_B = map(int, sys.stdin.readline().split())
    B = [sys.stdin.readline().rstrip() for _ in range(H_B)]
    H_X, W_X = map(int, sys.stdin.readline().split())
    X = [sys.stdin.readline().rstrip() for _ in range(H_X)]

    # Collect black positions in A and B
    blacks_A = [(i, j) for i in range(H_A) for j in range(W_A) if A[i][j] == '#']
    blacks_B = [(i, j) for i in range(H_B) for j in range(W_B) if B[i][j] == '#']

    # Find min/max row/col for A
    A_i_min = min(i for i, j in blacks_A)
    A_i_max = max(i for i, j in blacks_A)
    A_j_min = min(j for i, j in blacks_A)
    A_j_max = max(j for i, j in blacks_A)

    # Find min/max row/col for B
    B_i_min = min(i for i, j in blacks_B)
    B_i_max = max(i for i, j in blacks_B)
    B_j_min = min(j for i, j in blacks_B)
    B_j_max = max(j for i, j in blacks_B)

    # Compute allowed shift ranges for A relative to X
    da_y_min = -A_i_min
    da_y_max = (H_X - 1) - A_i_max
    da_x_min = -A_j_min
    da_x_max = (W_X - 1) - A_j_max

    # Compute allowed shift ranges for B relative to X
    db_y_min = -B_i_min
    db_y_max = (H_X - 1) - B_i_max
    db_x_min = -B_j_min
    db_x_max = (W_X - 1) - B_j_max

    # If either has no valid shift range, impossible
    if da_y_min > da_y_max or da_x_min > da_x_max or db_y_min > db_y_max or db_x_min > db_x_max:
        print("No")
        return

    # Try all shifts
    for da_y in range(da_y_min, da_y_max + 1):
        for da_x in range(da_x_min, da_x_max + 1):
            for db_y in range(db_y_min, db_y_max + 1):
                for db_x in range(db_x_min, db_x_max + 1):
                    # Build the overlaid H_X x W_X grid
                    # Start with all '.'
                    C = [list('.' * W_X) for _ in range(H_X)]
                    # Place A blacks
                    for i, j in blacks_A:
                        y = i + da_y
                        x = j + da_x
                        # These are guaranteed in-bounds by construction
                        C[y][x] = '#'
                    # Place B blacks
                    for i, j in blacks_B:
                        y = i + db_y
                        x = j + db_x
                        C[y][x] = '#'
                    # Compare to X
                    ok = True
                    for i in range(H_X):
                        # join C[i] back to string for fast compare
                        if ''.join(C[i]) != X[i]:
                            ok = False
                            break
                    if ok:
                        print("Yes")
                        return

    # If no combination worked
    print("No")


if __name__ == "__main__":
    main()