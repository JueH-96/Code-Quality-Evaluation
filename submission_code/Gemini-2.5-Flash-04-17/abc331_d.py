import sys

# Use fast I/O
input = sys.stdin.readline

def solve():
    N, Q = map(int, input().split())
    P = [input().strip() for _ in range(N)]

    # Create black_pattern: 1 if 'B', 0 if 'W'
    black_pattern = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if P[i][j] == 'B':
                black_pattern[i][j] = 1

    # Create 1-indexed prefix sum table for black_pattern
    # pps[i][j] = sum of black_pattern[row][col] for 0 <= row < i and 0 <= col < j
    # Size (N+1) x (N+1)
    pps = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            pps[i][j] = black_pattern[i-1][j-1] + pps[i-1][j] + pps[i][j-1] - pps[i-1][j-1]

    # Helper to get prefix sum from 1-indexed table for pattern rectangle [0, r] x [0, c]
    # r, c are 0-indexed pattern indices, range [0, N-1]
    def get_pattern_prefix_sum(r, c):
        # If r or c is negative, the sum over the pattern subgrid is 0
        if r < 0 or c < 0:
            return 0
        # Map 0-indexed pattern coords [0, r] x [0, c] to 1-indexed pps table coords [0+1, r+1] x [0+1, c+1]
        return pps[r+1][c+1]

    # Function to count black squares in the large grid rectangle (0, 0) to (R, C)
    # R, C are 0-indexed grid coordinates, range [0, 10^9 - 1] or potentially -1
    def count_black(R, C):
        # If R or C is negative, the rectangle is empty, count is 0
        if R < 0 or C < 0:
            return 0

        # Calculate div and mod for R and C with respect to N
        # R = R_div * N + R_mod, where 0 <= R_mod < N
        R_div = R // N
        R_mod = R % N
        C_div = C // N
        C_mod = C % N

        # Count black squares using the derived formula
        # The rectangle [0, R] x [0, C] can be divided into:
        # 1. A (R_div) x (C_div) block of full N*N patterns
        # 2. A (R_div) x 1 block of full N-row strips covering pattern columns [0, C_mod]
        # 3. A 1 x (C_div) block of full N-column strips covering pattern rows [0, R_mod]
        # 4. A 1 x 1 block covering pattern rows [0, R_mod] and columns [0, C_mod]

        # Term 1: Full N*N blocks
        # There are R_div full N-row blocks and C_div full N-column blocks.
        # Total (R_div) * (C_div) full N*N blocks. Each contains get_pattern_prefix_sum(N-1, N-1) black squares.
        total = R_div * C_div * get_pattern_prefix_sum(N-1, N-1)

        # Term 2: Full N-row blocks, partial column block
        # There are R_div full N-row blocks.
        # The columns covered are from (C_div * N) to C, which correspond to pattern columns 0 to C_mod.
        # For each of the R_div full N-row blocks, the black squares in pattern rectangle [0, N-1] x [0, C_mod] are counted once.
        total += R_div * get_pattern_prefix_sum(N-1, C_mod)

        # Term 3: Partial row block, full N-column blocks
        # There are C_div full N-column blocks.
        # The rows covered are from (R_div * N) to R, which correspond to pattern rows 0 to R_mod.
        # For each of the C_div full N-column blocks, the black squares in pattern rectangle [0, R_mod] x [0, N-1] are counted once.
        total += C_div * get_pattern_prefix_sum(R_mod, N-1)

        # Term 4: Partial row block, partial column block
        # The rows covered are from (R_div * N) to R, corresponding to pattern rows 0 to R_mod.
        # The columns covered are from (C_div * N) to C, corresponding to pattern columns 0 to C_mod.
        # The black squares in pattern rectangle [0, R_mod] x [0, C_mod] are counted once.
        total += get_pattern_prefix_sum(R_mod, C_mod)

        return total

    # Process queries
    # Read Q queries
    for _ in range(Q):
        # Each query is A, B, C, D, defining rectangle [A, C] x [B, D]
        # A, B, C, D are 0-indexed grid coordinates
        A, B, C, D = map(int, input().split())
        
        # Use 2D inclusion-exclusion to find the sum in the rectangle [A, C] x [B, D]
        # Sum(A,B to C,D) = Sum(0,0 to C,D) - Sum(0,0 to A-1,D) - Sum(0,0 to C,B-1) + Sum(0,0 to A-1,B-1)
        # The count_black(R, C) function calculates Sum(0,0 to R, C)
        
        ans = count_black(C, D) - count_black(A-1, D) - count_black(C, B-1) + count_black(A-1, B-1)
        print(ans)

solve()