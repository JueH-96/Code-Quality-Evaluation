class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        m, n = len(coins), len(coins[0])
        
        # dp[i][j][k] will store the maximum coins we can collect at cell (i, j)
        # with k neutralizations used
        dp = [[[-float('inf')] * 3 for _ in range(n)] for _ in range(m)]
        
        # Initialize the starting point
        dp[0][0][0] = coins[0][0]
        
        for i in range(m):
            for j in range(n):
                for k in range(3):
                    if i == 0 and j == 0:
                        continue
                    
                    current_value = coins[i][j]
                    
                    # Calculate the maximum coins we can have at (i, j) with k neutralizations
                    if i > 0:
                        # Coming from the top
                        for prev_k in range(k + 1):
                            if current_value < 0 and prev_k < 2:
                                dp[i][j][k] = max(dp[i][j][k], dp[i-1][j][prev_k] - current_value)
                            dp[i][j][k] = max(dp[i][j][k], dp[i-1][j][prev_k] + current_value)
                    
                    if j > 0:
                        # Coming from the left
                        for prev_k in range(k + 1):
                            if current_value < 0 and prev_k < 2:
                                dp[i][j][k] = max(dp[i][j][k], dp[i][j-1][prev_k] - current_value)
                            dp[i][j][k] = max(dp[i][j][k], dp[i][j-1][prev_k] + current_value)
        
        # The answer is the maximum coins we can collect at the bottom-right corner
        return max(dp[m-1][n-1])