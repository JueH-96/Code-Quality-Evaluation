def main():
    import sys
    import numpy as np

    data = sys.stdin.buffer.read().split()
    if not data:
        return
    # Parse input values
    it = iter(data)
    H = int(next(it))
    W = int(next(it))
    N = int(next(it))
    
    # Create a grid of size H x W with 1 indicating a safe (non-holed) cell.
    grid = np.ones((H, W), dtype=np.int32)
    
    # Mark the holed squares as 0.
    for _ in range(N):
        a = int(next(it)) - 1  # converting from 1-indexed to 0-indexed
        b = int(next(it)) - 1
        grid[a, b] = 0

    # We'll use the typical dynamic programming algorithm:
    #   dp[i][j] = 0 if grid[i][j] is holed, otherwise:
    #   dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1.
    # dp[i][j] is the maximum side length of a square ending at (i,j) (with (i,j) as bottom-right corner).
    # A square of side length n contributes n holeless squares ending at that cell (of sizes 1,2,...,n).
    
    # Initialize dp with grid values (first row and first column remain the same).
    dp = grid.copy()
    
    # Fill dp for rows 1 to H-1 using vectorized operations on each row.
    for i in range(1, H):
        # For columns from 1 to W-1:
        # Only compute if grid[i, col] is 1.
        dp[i, 1:] = grid[i, 1:] * (np.minimum(np.minimum(dp[i-1, 1:], dp[i, :-1]), dp[i-1, :-1]) + 1)
    
    # The answer is the sum of dp array.
    # Each cell with dp[i][j] = n adds n holeless squares (squares with side lengths 1 through n).
    ans = int(dp.sum())
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()