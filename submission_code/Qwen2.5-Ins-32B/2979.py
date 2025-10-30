from typing import List

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Sort offers based on the end time
        offers.sort(key=lambda x: x[1])
        dp = [0] * (n + 1)
        
        i = 0
        for j in range(1, n + 1):
            dp[j] = dp[j - 1]
            while i < len(offers) and offers[i][1] == j - 1:
                dp[j] = max(dp[j], dp[offers[i][0]] + offers[i][2])
                i += 1
        
        return dp[n]