from typing import List

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        offers.sort(key=lambda x: x[1])
        dp = [0] * (n + 1)
        j = 0
        for i in range(1, n + 1):
            dp[i] = dp[i - 1]
            while j < len(offers) and offers[j][1] == i - 1:
                dp[i] = max(dp[i], dp[offers[j][0]] + offers[j][2])
                j += 1
        return dp[-1]