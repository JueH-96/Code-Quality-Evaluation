def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    idx = 0
    # Read sheet A.
    H_A, W_A = map(int, data[idx].split())
    idx += 1
    A = data[idx: idx + H_A]
    idx += H_A

    # Read sheet B.
    H_B, W_B = map(int, data[idx].split())
    idx += 1
    B = data[idx: idx + H_B]
    idx += H_B

    # Read sheet X.
    H_X, W_X = map(int, data[idx].split())
    idx += 1
    X = data[idx: idx + H_X]
    idx += H_X

    # Get coordinates of black squares for each sheet.
    A_black = []
    for i in range(H_A):
        for j in range(W_A):
            if A[i][j] == '#':
                A_black.append((i, j))
    B_black = []
    for i in range(H_B):
        for j in range(W_B):
            if B[i][j] == '#':
                B_black.append((i, j))
    X_black = set()
    for i in range(H_X):
        for j in range(W_X):
            if X[i][j] == '#':
                X_black.add((i, j))

    # For a sheet with black squares, determine the min and max row/column indices.
    A_r_min = min(i for (i, j) in A_black)
    A_r_max = max(i for (i, j) in A_black)
    A_c_min = min(j for (i, j) in A_black)
    A_c_max = max(j for (i, j) in A_black)
    B_r_min = min(i for (i, j) in B_black)
    B_r_max = max(i for (i, j) in B_black)
    B_c_min = min(j for (i, j) in B_black)
    B_c_max = max(j for (i, j) in B_black)

    # When pasting sheets A and B onto the final cut-out area (of dimensions H_X x W_X),
    # the condition is that for every black square (i, j) in the sheet, the shifted coordinate
    # (offset + i, offset + j) must lie inside 0<=row<H_X and 0<=col<W_X.
    # Therefore, for sheet A, if we let (rA, cA) be the offset, then:
    #   rA + A_r_min >= 0   and   rA + A_r_max <= H_X - 1.
    # Similarly for the column.
    # We can therefore enumerate:
    #   rA in range(-A_r_min, H_X - A_r_max)
    #   cA in range(-A_c_min, W_X - A_c_max)
    # and similarly for sheet B.

    for rA in range(-A_r_min, H_X - A_r_max):
        for cA in range(-A_c_min, W_X - A_c_max):
            # Compute positions for sheet A's black squares after translation.
            posA = {(rA + i, cA + j) for (i, j) in A_black}
            for rB in range(-B_r_min, H_X - B_r_max):
                for cB in range(-B_c_min, W_X - B_c_max):
                    posB = {(rB + i, cB + j) for (i, j) in B_black}
                    # The painted positions are the union of A and B black squares.
                    painted = posA | posB
                    if painted == X_black:
                        print("Yes")
                        return

    print("No")


if __name__ == '__main__':
    main()