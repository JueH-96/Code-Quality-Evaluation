from typing import List

class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        m, n = len(coins), len(coins[0])
        
        # dp[i][j][k] will hold the maximum coins we can collect reaching (i, j) using k neutralizations
        dp = [[[-float('inf')] * 3 for _ in range(n)] for _ in range(m)]
        
        # Starting point
        dp[0][0][0] = coins[0][0]  # No neutralization used
        dp[0][0][1] = coins[0][0]  # One neutralization used
        dp[0][0][2] = coins[0][0]  # Two neutralizations used
        
        for i in range(m):
            for j in range(n):
                for k in range(3):  # k is the number of neutralizations used
                    if i == 0 and j == 0:
                        continue
                    
                    # Calculate the current cell value
                    current_value = coins[i][j]
                    
                    # If we move from the top (i-1, j)
                    if i > 0:
                        if current_value >= 0:
                            dp[i][j][k] = max(dp[i][j][k], dp[i-1][j][k] + current_value)
                        elif k > 0:  # If we can neutralize the robber
                            dp[i][j][k] = max(dp[i][j][k], dp[i-1][j][k-1] + 0)  # Neutralize and gain nothing
                    
                    # If we move from the left (i, j-1)
                    if j > 0:
                        if current_value >= 0:
                            dp[i][j][k] = max(dp[i][j][k], dp[i][j-1][k] + current_value)
                        elif k > 0:  # If we can neutralize the robber
                            dp[i][j][k] = max(dp[i][j][k], dp[i][j-1][k-1] + 0)  # Neutralize and gain nothing
        
        # The answer is the maximum coins we can collect at the bottom-right corner with any number of neutralizations
        return max(dp[m-1][n-1][0], dp[m-1][n-1][1], dp[m-1][n-1][2])