class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        m, n = len(coins), len(coins[0])
        
        # dp[i][j][k] = max coins when reaching (i,j) with exactly k neutralizations used
        dp = [[[-float('inf')] * 3 for _ in range(n)] for _ in range(m)]
        
        # Base case
        dp[0][0][0] = coins[0][0]
        if coins[0][0] < 0:
            dp[0][0][1] = 0  # neutralize the robber at (0,0)
        
        # Fill the dp table
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                
                # Coming from top
                if i > 0:
                    for k in range(3):
                        if dp[i-1][j][k] == -float('inf'):
                            continue
                        if coins[i][j] >= 0:
                            dp[i][j][k] = max(dp[i][j][k], dp[i-1][j][k] + coins[i][j])
                        else:
                            # Don't neutralize
                            dp[i][j][k] = max(dp[i][j][k], dp[i-1][j][k] + coins[i][j])
                            # Neutralize (if possible)
                            if k > 0:
                                dp[i][j][k] = max(dp[i][j][k], dp[i-1][j][k-1] + 0)
                
                # Coming from left
                if j > 0:
                    for k in range(3):
                        if dp[i][j-1][k] == -float('inf'):
                            continue
                        if coins[i][j] >= 0:
                            dp[i][j][k] = max(dp[i][j][k], dp[i][j-1][k] + coins[i][j])
                        else:
                            # Don't neutralize
                            dp[i][j][k] = max(dp[i][j][k], dp[i][j-1][k] + coins[i][j])
                            # Neutralize (if possible)
                            if k > 0:
                                dp[i][j][k] = max(dp[i][j][k], dp[i][j-1][k-1] + 0)
        
        # Return the maximum among all possible neutralizations used
        result = -float('inf')
        for k in range(3):
            result = max(result, dp[m-1][n-1][k])
        return result