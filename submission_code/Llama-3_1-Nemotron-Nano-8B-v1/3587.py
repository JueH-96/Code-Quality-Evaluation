from typing import List

class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        # Initialize DP table with k rows and n columns
        dp = [[0] * n for _ in range(k)]
        
        # Compute the first day (i=0)
        for j in range(n):
            max_move = 0
            for m in range(n):
                if m != j:
                    if travelScore[m][j] > max_move:
                        max_move = travelScore[m][j]
            dp[0][j] = max(stayScore[0][j], max_move)
        
        # Compute days from 1 to k-1
        for i in range(1, k):
            for j in range(n):
                # Option 1: Stay in city j
                option1 = dp[i-1][j] + stayScore[i][j]
                
                # Option 2: Move from any city m to j (m != j)
                option2 = 0
                for m in range(n):
                    if m != j:
                        current = dp[i-1][m] + travelScore[m][j]
                        if current > option2:
                            option2 = current
                
                dp[i][j] = max(option1, option2)
        
        # The maximum score after k days is the maximum in the last row
        return max(dp[k-1])