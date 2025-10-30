class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        m, n = len(coins), len(coins[0])
        
        # dp[i][j][k] = maximum coins at position (i,j) with k neutralizations used
        # k can be 0, 1, or 2
        dp = [[[-float('inf')] * 3 for _ in range(n)] for _ in range(m)]
        
        # Base case: starting position
        if coins[0][0] >= 0:
            dp[0][0][0] = coins[0][0]
        else:
            # Can start by neutralizing the first robber
            dp[0][0][0] = coins[0][0]  # Take the loss
            dp[0][0][1] = 0  # Neutralize
        
        # Fill the DP table
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                
                # For each number of neutralizations used
                for k in range(3):
                    # Coming from top (i-1, j)
                    if i > 0 and dp[i-1][j][k] != -float('inf'):
                        if coins[i][j] >= 0:
                            # Gain coins
                            dp[i][j][k] = max(dp[i][j][k], dp[i-1][j][k] + coins[i][j])
                        else:
                            # Robber cell - two choices
                            # 1. Take the loss
                            dp[i][j][k] = max(dp[i][j][k], dp[i-1][j][k] + coins[i][j])
                            # 2. Neutralize if we have neutralizations left
                            if k < 2 and dp[i-1][j][k] != -float('inf'):
                                dp[i][j][k+1] = max(dp[i][j][k+1], dp[i-1][j][k])
                    
                    # Coming from left (i, j-1)
                    if j > 0 and dp[i][j-1][k] != -float('inf'):
                        if coins[i][j] >= 0:
                            # Gain coins
                            dp[i][j][k] = max(dp[i][j][k], dp[i][j-1][k] + coins[i][j])
                        else:
                            # Robber cell - two choices
                            # 1. Take the loss
                            dp[i][j][k] = max(dp[i][j][k], dp[i][j-1][k] + coins[i][j])
                            # 2. Neutralize if we have neutralizations left
                            if k < 2 and dp[i][j-1][k] != -float('inf'):
                                dp[i][j][k+1] = max(dp[i][j][k+1], dp[i][j-1][k])
        
        # Return the maximum coins at destination with any number of neutralizations
        return max(dp[m-1][n-1][0], dp[m-1][n-1][1], dp[m-1][n-1][2])