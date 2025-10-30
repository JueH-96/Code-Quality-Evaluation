def main():
    import sys

    # Read input
    H_A, W_A = map(int, sys.stdin.readline().split())
    A = [sys.stdin.readline().strip() for _ in range(H_A)]
    H_B, W_B = map(int, sys.stdin.readline().split())
    B = [sys.stdin.readline().strip() for _ in range(H_B)]
    H_X, W_X = map(int, sys.stdin.readline().split())
    X = [sys.stdin.readline().strip() for _ in range(H_X)]

    # Collect black squares for A and B
    A_black = [(i, j) for i in range(H_A) for j in range(W_A) if A[i][j] == '#']
    B_black = [(i, j) for i in range(H_B) for j in range(W_B) if B[i][j] == '#']

    # Compute ranges for A's positions
    a_row_low = max(-i for (i, j) in A_black)
    a_row_high = min((H_X - 1 - i) for (i, j) in A_black)
    a_col_low = max(-j for (i, j) in A_black)
    a_col_high = min((W_X - 1 - j) for (i, j) in A_black)

    # Compute ranges for B's positions
    b_row_low = max(-i for (i, j) in B_black)
    b_row_high = min((H_X - 1 - i) for (i, j) in B_black)
    b_col_low = max(-j for (i, j) in B_black)
    b_col_high = min((W_X - 1 - j) for (i, j) in B_black)

    found = False

    # Iterate over all possible positions for A and B
    for a_row in range(a_row_low, a_row_high + 1):
        for a_col in range(a_col_low, a_col_high + 1):
            for b_row in range(b_row_low, b_row_high + 1):
                for b_col in range(b_col_low, b_col_high + 1):
                    valid = True
                    for x_row in range(H_X):
                        for x_col in range(W_X):
                            # Check A's contribution
                            a_i = x_row - a_row
                            a_j = x_col - a_col
                            a_has = False
                            if 0 <= a_i < H_A and 0 <= a_j < W_A:
                                if A[a_i][a_j] == '#':
                                    a_has = True
                            # Check B's contribution
                            b_i = x_row - b_row
                            b_j = x_col - b_col
                            b_has = False
                            if 0 <= b_i < H_B and 0 <= b_j < W_B:
                                if B[b_i][b_j] == '#':
                                    b_has = True
                            # Compare with X's cell
                            expected = X[x_row][x_col]
                            if (a_has or b_has) and expected != '#':
                                valid = False
                                break
                            elif not (a_has or b_has) and expected == '#':
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        found = True
                        break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if found:
            break

    print("Yes" if found else "No")

if __name__ == "__main__":
    main()