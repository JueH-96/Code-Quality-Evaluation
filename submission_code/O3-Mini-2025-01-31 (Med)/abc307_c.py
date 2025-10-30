def main():
    import sys

    data = sys.stdin.read().splitlines()
    if not data: 
        return
    idx = 0
    # Read sheet A
    H_A, W_A = map(int, data[idx].split())
    idx += 1
    A = []
    for _ in range(H_A):
        A.append(data[idx])
        idx += 1

    # Read sheet B
    H_B, W_B = map(int, data[idx].split())
    idx += 1
    B = []
    for _ in range(H_B):
        B.append(data[idx])
        idx += 1

    # Read sheet X (target)
    H_X, W_X = map(int, data[idx].split())
    idx += 1
    X = []
    for _ in range(H_X):
        X.append(data[idx])
        idx += 1

    # We want to decide if we can choose placements for sheet A and B when pasting
    # them onto an infinite transparent board, and then cut an H_X x W_X rectangle
    # that covers all pasted black squares ( '#' ) such that after overlapping
    # (black squares are ORed), that rectangle exactly equals the given pattern X.
    # The constraint “The cut-out sheet includes all black squares of sheets A and B”
    # means that every black square (i.e. every '#' in A and B) must fall inside the chosen rectangle.
    #
    # For convenience, we assume our final board (the cut-out) is exactly the grid for X;
    # i.e. a grid of size H_X x W_X. Then we want to paste A and B using translations
    # such that every black square in A and B lands inside the grid and the union (OR)
    # equals X.
    #
    # We can choose translations arbitrarily. Let (r, c) represent the offset at which
    # we paste the top-left corner of a sheet relative to the top-left of the cut-out grid.
    # Since the sheet may be pasted in such a way that if some part is not black,
    # it can lie outside X. However, since each sheet must have all its black cells
    # inside the cut-out grid, for sheet A, for each cell (i, j) with A[i][j]=='#',
    # we must have:
    #      0 <= r_A + i < H_X and 0 <= c_A + j < W_X.
    #
    # Because H_A, W_A, etc. are small (<= 10) we can try all possible placements:
    # For sheet A: try all (r_A, c_A) with r_A in range(-H_A+1, H_X) and c_A in range(-W_A+1, W_X)
    # that satisfy the condition above for all '#' in A. Do similarly for B.

    validA_translations = []
    for ra in range(-H_A+1, H_X):
        for ca in range(-W_A+1, W_X):
            valid = True
            for i in range(H_A):
                for j in range(W_A):
                    if A[i][j] == '#':
                        R = ra + i
                        C = ca + j
                        if not (0 <= R < H_X and 0 <= C < W_X):
                            valid = False
                            break
                if not valid:
                    break
            if valid:
                validA_translations.append((ra, ca))

    validB_translations = []
    for rb in range(-H_B+1, H_X):
        for cb in range(-W_B+1, W_X):
            valid = True
            for i in range(H_B):
                for j in range(W_B):
                    if B[i][j] == '#':
                        R = rb + i
                        C = cb + j
                        if not (0 <= R < H_X and 0 <= C < W_X):
                            valid = False
                            break
                if not valid:
                    break
            if valid:
                validB_translations.append((rb, cb))

    # Try every combination of valid placements for A and B.
    for ra, ca in validA_translations:
        for rb, cb in validB_translations:
            # Create a grid for the final pasted board (of dimensions H_X x W_X),
            # initially all transparent ('.').
            grid = [['.' for _ in range(W_X)] for _ in range(H_X)]
            # Paste sheet A.
            for i in range(H_A):
                for j in range(W_A):
                    if A[i][j] == '#':
                        grid[ra + i][ca + j] = '#'
            # Paste sheet B (overlapping black squares remain black).
            for i in range(H_B):
                for j in range(W_B):
                    if B[i][j] == '#':
                        grid[rb + i][cb + j] = '#'
            # Check if the resulting grid equals X.
            ok = True
            for i in range(H_X):
                if "".join(grid[i]) != X[i]:
                    ok = False
                    break
            if ok:
                sys.stdout.write("Yes")
                return

    sys.stdout.write("No")


if __name__ == '__main__':
    main()