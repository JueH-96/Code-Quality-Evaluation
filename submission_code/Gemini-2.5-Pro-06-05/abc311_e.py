import sys

def solve():
    """
    Solves the Holeless Squares problem using memory-optimized dynamic programming.
    """
    # Use fast I/O, which is a good practice for competitive programming in Python.
    input = sys.stdin.readline

    # Read grid dimensions H, W and the number of holes N.
    try:
        H, W, N = map(int, input().split())
    except ValueError:
        # This handles cases where the input might end with an empty line.
        return

    # Read the coordinates of N holed squares.
    # We use a set for efficient O(1) average time complexity lookups.
    # The input coordinates are 1-based, so we convert them to 0-based.
    holes = set()
    for _ in range(N):
        a, b = map(int, input().split())
        holes.add((a - 1, b - 1))

    # dp_prev_row stores the DP values for the row r+1.
    # We use a size of W+1 to have a zero-padded boundary, which simplifies
    # the logic inside the loops by avoiding explicit boundary checks.
    dp_prev_row = [0] * (W + 1)
    
    total_holeless_squares = 0

    # We iterate through the rows from bottom to top (from H-1 down to 0).
    for r in range(H - 1, -1, -1):
        # dp_curr_row will store the DP values for the current row `r`.
        dp_curr_row = [0] * (W + 1)
        
        # We iterate through the columns from right to left (from W-1 down to 0).
        for c in range(W - 1, -1, -1):
            # If the current cell (r, c) is not a hole:
            if (r, c) not in holes:
                # Calculate dp[r][c] using the recurrence relation.
                # dp_prev_row[c] corresponds to dp[r+1][c]
                # dp_curr_row[c+1] corresponds to dp[r][c+1]
                # dp_prev_row[c+1] corresponds to dp[r+1][c+1]
                dp_curr_row[c] = 1 + min(dp_prev_row[c], dp_curr_row[c + 1], dp_prev_row[c + 1])
            
            # If (r, c) is a hole, dp_curr_row[c] remains 0 as initialized.
            
            # The value dp_curr_row[c] represents the side length of the largest holeless
            # square starting at (r, c). This is also the number of holeless squares
            # with (r,c) as their top-left corner. We add this to our total.
            total_holeless_squares += dp_curr_row[c]
            
        # The current row's results become the previous row's results for the next iteration.
        dp_prev_row = dp_curr_row

    # Print the final total count.
    print(total_holeless_squares)

solve()