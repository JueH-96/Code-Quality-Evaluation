from typing import List

class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Sort the cuts in descending order to prioritize higher costs first
        horizontalCut.sort(reverse=True)
        verticalCut.sort(reverse=True)
        
        INF = float('inf')
        # Initialize DP table with infinity
        dp = [[INF] * n for _ in range(m)]
        dp[0][0] = 0  # Base case: no cuts made
        
        for h in range(m):
            for v in range(n):
                if h == 0 and v == 0:
                    continue  # Skip the base case
                current = INF
                # Check if we can take a horizontal cut
                if h > 0:
                    current = min(current, dp[h-1][v] + horizontalCut[h-1] * (v + 1))
                # Check if we can take a vertical cut
                if v > 0:
                    current = min(current, dp[h][v-1] + verticalCut[v-1] * (h + 1))
                dp[h][v] = current
        
        return dp[m-1][n-1]