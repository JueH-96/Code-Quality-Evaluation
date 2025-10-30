import sys

def solve():
    line = sys.stdin.readline().split()
    H = int(line[0])
    W = int(line[1])
    N = int(line[2])

    holed_cells = set()
    for _ in range(N):
        a, b = map(int, sys.stdin.readline().split())
        holed_cells.add((a, b))

    # dp[r][c] stores the maximum size of a holeless square
    # whose bottom-right corner is (r, c).
    # We use space optimization, keeping only the previous row's DP values.
    # prev_dp[c] stores dp[r-1][c]
    # curr_dp[c] stores dp[r][c]
    
    # Initialize dp table for the conceptual row 0 (all zeros)
    prev_dp = [0] * (W + 1)
    total_holeless_squares = 0

    for r in range(1, H + 1):
        # Initialize dp table for the current row r
        curr_dp = [0] * (W + 1)
        
        for c in range(1, W + 1):
            if (r, c) in holed_cells:
                curr_dp[c] = 0
            else:
                # The maximum size of a square of '1's ending at (r, c)
                # in a binary matrix where non-holed cells are 1 and holed cells are 0.
                # This is a standard DP relation.
                # dp[r][c] = 1 + min(dp[r-1][c], dp[r][c-1], dp[r-1][c-1])
                # Using the space-optimized representation:
                # dp[r-1][c] is prev_dp[c]
                # dp[r][c-1] is curr_dp[c-1] (already computed in the current row)
                # dp[r-1][c-1] is prev_dp[c-1]
                curr_dp[c] = 1 + min(prev_dp[c], curr_dp[c-1], prev_dp[c-1])
                
            total_holeless_squares += curr_dp[c]

        # The current row becomes the previous row for the next iteration
        # Assigning curr_dp to prev_dp updates the reference.
        prev_dp = curr_dp

    print(total_holeless_squares)

solve()