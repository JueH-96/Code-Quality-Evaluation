from typing import List

class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        dp = [[0] * n for _ in range(k)]
        
        # Initialize the first day
        for j in range(n):
            max_travel = max(travelScore[m][j] for m in range(n))
            dp[0][j] = max(stayScore[0][j], max_travel)
        
        # Fill in the rest of the days
        for i in range(1, k):
            for j in range(n):
                option1 = dp[i-1][j] + stayScore[i][j]
                option2 = max(dp[i-1][m] + travelScore[m][j] for m in range(n))
                dp[i][j] = max(option1, option2)
        
        return max(dp[-1])