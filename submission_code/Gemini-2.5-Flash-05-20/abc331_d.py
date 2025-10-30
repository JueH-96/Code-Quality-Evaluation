import sys

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    P = [sys.stdin.readline().strip() for _ in range(N)]

    # Precompute 2D prefix sums for the N x N pattern
    # pref[r][c] will store the number of black squares in P[0...r-1][0...c-1]
    # So, pref will be of size (N+1) x (N+1) to handle 1-based indexing for sums.
    pref = [[0] * (N + 1) for _ in range(N + 1)]

    for i in range(N):
        for j in range(N):
            is_black = 1 if P[i][j] == 'B' else 0
            # Calculate pref[i+1][j+1] using values from adjacent cells
            pref[i+1][j+1] = pref[i][j+1] + pref[i+1][j] - pref[i][j] + is_black

    # Helper function to calculate black squares in rectangle (0,0) to (R,C)
    # R, C are 0-indexed coordinates of the bottom-right corner.
    def get_black_count(R, C):
        # If R or C is negative, the rectangle is empty, so count is 0.
        if R < 0 or C < 0:
            return 0

        # Number of rows and columns in the requested rectangle (R+1 rows, C+1 columns)
        rows_in_rect = R + 1
        cols_in_rect = C + 1

        # Calculate how many full N x N blocks fit into the total rows/columns
        r_full_blocks = rows_in_rect // N
        c_full_blocks = cols_in_rect // N

        # Calculate the dimensions of the partial block (remaining rows/columns)
        r_partial_rows = rows_in_rect % N
        c_partial_cols = cols_in_rect % N

        # The total count is sum of four parts:
        count = (
            # 1. Black squares in all full N x N blocks
            r_full_blocks * c_full_blocks * pref[N][N]
            # 2. Black squares in full N-row strips, with partial columns
            + r_full_blocks * pref[N][c_partial_cols]
            # 3. Black squares in partial-row strips, with full N columns
            + c_full_blocks * pref[r_partial_rows][N]
            # 4. Black squares in the remaining partial row and partial column block
            + pref[r_partial_rows][c_partial_cols]
        )
        return count

    results = []
    for _ in range(Q):
        A, B, C, D = map(int, sys.stdin.readline().split())

        # Use 2D inclusion-exclusion principle to find black squares in (A,B) to (C,D)
        # Sum(C,D) - Sum(A-1,D) - Sum(C,B-1) + Sum(A-1,B-1)
        ans = (
            get_black_count(C, D)
            - get_black_count(A - 1, D)
            - get_black_count(C, B - 1)
            + get_black_count(A - 1, B - 1)
        )
        results.append(str(ans))
    
    # Print all results separated by newlines
    sys.stdout.write("
".join(results) + "
")

solve()