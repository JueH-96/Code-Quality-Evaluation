from typing import List

class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Initialize dp table with infinity
        dp = [[float('inf')] * n for _ in range(m)]
        dp[0][0] = 0  # Starting point with no cuts made
        
        # Fill the dp table
        for i in range(m):
            for j in range(n):
                if dp[i][j] != float('inf'):
                    # Make a horizontal cut if possible
                    if i < m - 1:
                        dp[i + 1][j] = min(dp[i + 1][j], dp[i][j] + (j + 1) * horizontalCut[i])
                    # Make a vertical cut if possible
                    if j < n - 1:
                        dp[i][j + 1] = min(dp[i][j + 1], dp[i][j] + (i + 1) * verticalCut[j])
        
        return dp[m - 1][n - 1]