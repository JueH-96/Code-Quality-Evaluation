# # YOUR CODE HERE - Start

import sys

# Use sys.stdin.readline for faster input reading, helpful for large inputs
# This is generally recommended in competitive programming for Python
input = sys.stdin.readline 

def solve():
    # Read grid dimensions H (height/rows), W (width/columns) and the number of holes N
    H, W, N = map(int, input().split())
    
    # Store hole coordinates in a set for efficient lookup. 
    # Checking membership in a set takes O(1) time on average.
    holes = set()
    for _ in range(N):
        # Read hole coordinates (r, c). The problem uses 1-based indexing.
        r, c = map(int, input().split())
        # Add the tuple (row, col) representing the hole location to the set.
        holes.add((r, c))

    # Initialize a DP table (dynamic programming table) of size (H+1) x (W+1) with zeros.
    # dp[i][j] will store the side length of the largest holeless square 
    # whose bottom-right corner is at the grid cell (i, j).
    # We use 1-based indexing for rows (1 to H) and columns (1 to W) to match the problem statement.
    # The 0-th row and 0-th column act as padding, which simplifies the boundary checks 
    # needed for the DP recurrence relation. They will always contain 0.
    dp = [[0] * (W + 1) for _ in range(H + 1)]
    
    # Variable to accumulate the total count of holeless squares.
    # Python integers support arbitrary precision, so we don't need to worry about overflow 
    # even if the total count becomes very large (like in Sample 4).
    total_holeless_squares = 0

    # Iterate through each cell of the grid, from top-left (1, 1) to bottom-right (H, W).
    for i in range(1, H + 1):
        for j in range(1, W + 1):
            # Check if the current cell (i, j) contains a hole.
            if (i, j) in holes:
                # If the cell (i, j) is a hole, then no square ending at (i, j) can be holeless
                # (except potentially a 0x0 square, which isn't counted). 
                # The largest side length is 0. Since dp[i][j] is initialized to 0,
                # we don't need to do anything here. The value remains 0.
                pass 
            else:
                # If the cell (i, j) is not a hole, we can potentially form a holeless square ending here.
                # The side length of the largest holeless square ending at (i, j) is determined by
                # the sizes of the largest holeless squares ending at its neighbors:
                # top neighbor (i-1, j), left neighbor (i, j-1), and top-left neighbor (i-1, j-1).
                # Specifically, it's 1 + the minimum of those three values. This is a standard DP recurrence relation
                # used for finding the largest square submatrix of 1s in a binary matrix. Here, non-holed squares act like 1s.
                # The padding ensures that if i=1 or j=1, the neighbors outside the grid boundary
                # correctly contribute a value of 0 (from dp[0][j], dp[i][0], or dp[0][0]).
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
            
            # The value dp[i][j] represents not just the side length of the largest square, but also
            # the total number of holeless squares that have (i, j) as their bottom-right corner.
            # For example, if dp[i][j] = k, it means squares of size 1x1, 2x2, ..., kxk ending at (i, j) 
            # are all holeless. There are k such squares.
            # Therefore, we add dp[i][j] to our running total count of all holeless squares.
            total_holeless_squares += dp[i][j]

    # After iterating through all grid cells, total_holeless_squares holds the final answer.
    # Print the total count to standard output.
    print(total_holeless_squares)

# Execute the main solution function. This starts the process of reading input and computing the answer.
solve()

# # YOUR CODE HERE - End