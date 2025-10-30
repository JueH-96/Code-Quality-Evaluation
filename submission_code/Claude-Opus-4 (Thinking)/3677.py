class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        m, n = len(coins), len(coins[0])
        
        # dp[i][j][k] = max coins at (i, j) with k neutralizations used
        dp = [[[-float('inf')] * 3 for _ in range(n)] for _ in range(m)]
        
        # Base case: starting position
        dp[0][0][0] = coins[0][0]
        if coins[0][0] < 0:
            dp[0][0][1] = 0  # neutralize the starting cell
        
        # Fill the DP table
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                
                for k in range(3):
                    # Coming from above
                    if i > 0:
                        if dp[i-1][j][k] != -float('inf'):
                            dp[i][j][k] = max(dp[i][j][k], dp[i-1][j][k] + coins[i][j])
                        
                        if coins[i][j] < 0 and k > 0 and dp[i-1][j][k-1] != -float('inf'):
                            dp[i][j][k] = max(dp[i][j][k], dp[i-1][j][k-1])
                    
                    # Coming from left
                    if j > 0:
                        if dp[i][j-1][k] != -float('inf'):
                            dp[i][j][k] = max(dp[i][j][k], dp[i][j-1][k] + coins[i][j])
                        
                        if coins[i][j] < 0 and k > 0 and dp[i][j-1][k-1] != -float('inf'):
                            dp[i][j][k] = max(dp[i][j][k], dp[i][j-1][k-1])
        
        return max(dp[m-1][n-1])