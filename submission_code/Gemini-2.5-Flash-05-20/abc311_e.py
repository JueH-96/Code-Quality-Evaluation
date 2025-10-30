import sys

def solve():
    H, W, N = map(int, sys.stdin.readline().split())

    # grid[r][c] will be 1 if (r, c) is holed, 0 otherwise.
    # Dimensions H+2 and W+2 are used to accommodate 1-based indexing
    # for the grid (from 1 to H and 1 to W) and to provide zero-initialized
    # padding for out-of-bounds accesses during DP calculation (e.g., dp[H+1][j] or dp[i][W+1]).
    # These padding cells effectively act as boundaries where no holeless square can extend.
    grid = [[0] * (W + 2) for _ in range(H + 2)]

    # Read the coordinates of holed squares and mark them in the grid.
    for _ in range(N):
        r, c = map(int, sys.stdin.readline().split())
        grid[r][c] = 1

    # dp[r][c] stores the side length 'n' of the largest holeless square
    # whose top-left corner is (r, c).
    # Initialize with zeros.
    dp = [[0] * (W + 2) for _ in range(H + 2)]

    # This variable will accumulate the total count of holeless squares.
    total_holeless_squares = 0

    # Iterate through the grid cells from bottom-right (H, W) to top-left (1, 1).
    # This reverse order ensures that when we calculate dp[r][c], the values
    # dp[r+1][c], dp[r][c+1], and dp[r+1][c+1] (which are needed for the recurrence)
    # have already been computed.
    for r in range(H, 0, -1):
        for c in range(W, 0, -1):
            if grid[r][c] == 1:
                # If the current square (r, c) is holed, it cannot be the
                # top-left corner of any holeless square of size greater than 0.
                # Thus, the maximum side length for a holeless square starting here is 0.
                dp[r][c] = 0
            else:
                # If the current square (r, c) is not holed, it can potentially
                # be the top-left corner of a holeless square.
                # The size of the largest holeless square with (r,c) as top-left
                # is 1 (for the square (r,c) itself) plus the minimum of the
                # maximum side lengths from its three immediate neighbors:
                # (r+1, c) (below), (r, c+1) (right), and (r+1, c+1) (bottom-right).
                # This recurrence correctly finds the largest square of non-holed cells.
                dp[r][c] = 1 + min(dp[r+1][c], dp[r][c+1], dp[r+1][c+1])
            
            # Each dp[r][c] value represents the maximum side length 'n'
            # for a holeless square with top-left (r,c).
            # If dp[r][c] = k, it means that squares of size 1, 2, ..., up to k
            # with top-left (r,c) are all holeless.
            # Thus, each dp[r][c] value directly contributes 'dp[r][c]' distinct
            # holeless squares (differing by size) to the total count.
            total_holeless_squares += dp[r][c]

    # Print the final total count of holeless squares.
    print(total_holeless_squares)

# Call the solve function to execute the program.
solve()