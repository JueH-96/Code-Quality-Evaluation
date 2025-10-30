def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    it = iter(input_data)
    H = int(next(it))
    W = int(next(it))
    N = int(next(it))
    
    # Create a 2D grid of booleans indicating whether the cell is holed.
    holed = [[False] * W for _ in range(H)]
    for _ in range(N):
        a = int(next(it)) - 1  # convert 1-indexed to 0-indexed
        b = int(next(it)) - 1
        holed[a][b] = True
        
    # We use dynamic programming to compute the maximum side length of a square with
    # bottom-right corner at (i, j) that does not contain any hole.
    # dp[i][j] = 0 if cell(i,j) is holed, else = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    # To ease indexing for i-1 and j-1 we use an extra row and column
    dp = [[0] * (W + 1) for _ in range(H + 1)]
    
    total = 0
    for i in range(1, H + 1):
        for j in range(1, W + 1):
            if holed[i-1][j-1]:
                dp[i][j] = 0
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            total += dp[i][j]
    
    sys.stdout.write(str(total))
    
if __name__ == '__main__':
    main()