from typing import List

class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Initialize the DP table
        dp = [[0] * n for _ in range(m)]
        
        # Base case: no cuts
        dp[0][0] = 0
        
        # Fill the first row: only vertical cuts
        for v in range(1, n):
            dp[0][v] = dp[0][v-1] + verticalCut[v-1] * 1  # Only one horizontal piece when h=0
        
        # Fill the first column: only horizontal cuts
        for h in range(1, m):
            dp[h][0] = dp[h-1][0] + horizontalCut[h-1] * 1  # Only one vertical piece when v=0
        
        # Fill the rest of the DP table
        for h in range(1, m):
            for v in range(1, n):
                # Cost if the last cut is horizontal
                cost_h_last = dp[h-1][v] + horizontalCut[h-1] * (v + 1)
                # Cost if the last cut is vertical
                cost_v_last = dp[h][v-1] + verticalCut[v-1] * (h + 1)
                # Choose the minimum cost
                dp[h][v] = min(cost_h_last, cost_v_last)
        
        return dp[m-1][n-1]