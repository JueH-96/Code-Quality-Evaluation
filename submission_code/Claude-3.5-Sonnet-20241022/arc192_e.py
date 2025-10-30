def solve():
    # Read input
    W, H, L, R, D, U = map(int, input().split())
    MOD = 998244353
    
    # Function to check if a point has a block
    def has_block(x, y):
        return x < L or x > R or y < D or y > U
    
    # dp[i][j] represents number of paths ending at point (i,j)
    dp = [[0] * (H + 1) for _ in range(W + 1)]
    
    # Initialize starting points
    for i in range(W + 1):
        for j in range(H + 1):
            if has_block(i, j):
                dp[i][j] = 1
    
    # For each point, add paths from left and bottom
    for i in range(W + 1):
        for j in range(H + 1):
            if has_block(i, j):
                # If we can come from left
                if i > 0:
                    dp[i][j] = (dp[i][j] + dp[i-1][j]) % MOD
                # If we can come from bottom
                if j > 0:
                    dp[i][j] = (dp[i][j] + dp[i][j-1]) % MOD
    
    # Sum up all possible paths
    result = 0
    for i in range(W + 1):
        for j in range(H + 1):
            if has_block(i, j):
                result = (result + dp[i][j]) % MOD
    
    print(result)

solve()