class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        m, n = len(coins), len(coins[0])
        dp = [[[0, 0] for _ in range(n)] for _ in range(m)]
        
        # Initialize the first row and column
        dp[0][0][0] = coins[0][0]
        for i in range(1, m):
            dp[i][0][0] = dp[i-1][0][0] + coins[i][0]
        for j in range(1, n):
            dp[0][j][0] = dp[0][j-1][0] + coins[0][j]
        
        # Fill the rest of the dp table
        for i in range(1, m):
            for j in range(1, n):
                # If the current cell has a robber, we can neutralize it
                if coins[i][j] < 0:
                    dp[i][j][0] = max(dp[i-1][j][0] + coins[i][j], dp[i][j-1][0] + coins[i][j])
                    dp[i][j][1] = max(dp[i-1][j][1] + coins[i][j], dp[i][j-1][1] + coins[i][j], dp[i-1][j-1][0] - coins[i][j])
                # If the current cell doesn't have a robber, we can't neutralize anything
                else:
                    dp[i][j][0] = max(dp[i-1][j][0], dp[i][j-1][0]) + coins[i][j]
                    dp[i][j][1] = max(dp[i-1][j][1], dp[i][j-1][1], dp[i-1][j-1][0])
        
        return max(dp[m-1][n-1][0], dp[m-1][n-1][1])