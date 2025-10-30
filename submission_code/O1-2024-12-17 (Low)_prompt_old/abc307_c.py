def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Parse inputs
    idx = 0
    H_A, W_A = map(int, input_data[idx:idx+2]); idx += 2
    A = input_data[idx:idx+H_A]; idx += H_A
    H_B, W_B = map(int, input_data[idx:idx+2]); idx += 2
    B = input_data[idx:idx+H_B]; idx += H_B
    H_X, W_X = map(int, input_data[idx:idx+2]); idx += 2
    X = input_data[idx:idx+H_X]; idx += H_X

    # Convert A, B, X into lists of black-square coordinates and also boolean grids
    A_blacks = []
    for r in range(H_A):
        for c in range(W_A):
            if A[r][c] == '#':
                A_blacks.append((r, c))
    B_blacks = []
    for r in range(H_B):
        for c in range(W_B):
            if B[r][c] == '#':
                B_blacks.append((r, c))
    X_bool = [[(X[r][c] == '#') for c in range(W_X)] for r in range(H_X)]

    # If there's only one black square in either A or B, it must at least fit
    # but the constraints guarantee at least one black square in each, so no special check needed.

    # We'll brute force all relative placements of B to A within a small range.
    # Reasoning: H_x, W_x <= 10, so bounding rectangle can be at most 10x10.
    # We'll let dr, dc range in about -20..20 just to be safe (this is still manageable).
    # Then compute the bounding rectangle that encloses A and B. If it is bigger than H_X x W_X, skip.
    # Otherwise, try placing that bounding rectangle somewhere in the 0..H_X-1 x 0..W_X-1 region.
    # Then see if it matches X exactly.

    # Function to get bounding rectangle for A at (0,0) and B at (dr,dc).
    def bounding_rect(dr, dc):
        top = min(0, dr)
        left = min(0, dc)
        bottom = max(H_A - 1, dr + H_B - 1)
        right = max(W_A - 1, dc + W_B - 1)
        return top, left, bottom, right

    # For checking a particular offset placement of the bounding rectangle in X.
    # R,C is the top-left corner in X (where bounding_rect_top,left is mapped).
    def check_match(dr, dc, R, C, top, left):
        # Build a union array of booleans for the bounding rectangle
        bh = (bottom - top + 1)
        bw = (right - left + 1)
        union_black = [[False]*bw for _ in range(bh)]

        # Mark squares from A
        for (ar, ac) in A_blacks:
            rr = ar - top
            cc = ac - left
            union_black[rr][cc] = True

        # Mark squares from B
        for (br, bc) in B_blacks:
            rr = (br + dr) - top
            cc = (bc + dc) - left
            union_black[rr][cc] = True

        # Now check against X_bool
        for rr in range(bh):
            for cc in range(bw):
                # The corresponding cell in X is at (R+rr, C+cc)
                if union_black[rr][cc] != X_bool[R+rr][C+cc]:
                    return False
        return True

    # Try all dr,dc in a suitable range
    # We'll pick a range that comfortably covers positions such that bounding rect won't exceed 10x10
    # In worst case, we can do -20..20. That is safe enough (41x41 = 1681) for dr,dc combined with other loops still fine.
    for dr in range(-20, 21):
        for dc in range(-20, 21):
            top, left, bottom, right = bounding_rect(dr, dc)
            bh = bottom - top + 1
            bw = right - left + 1
            # If bounding rect doesn't fit into H_X x W_X, skip
            if bh > H_X or bw > W_X:
                continue
            # Now try placing that bounding rectangle inside X in all possible ways
            # so that (top,left) aligns with (R,C) in X, i.e. bounding rect corners map to X[R..R+bh-1][C..C+bw-1]
            for R in range(H_X - bh + 1):
                for C in range(W_X - bw + 1):
                    if check_match(dr, dc, R, C, top, left):
                        print("Yes")
                        return
    print("No")

def __main():
    solve()