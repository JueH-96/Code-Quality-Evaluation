from typing import List

class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        dp = [[float('inf')] * 501 for _ in range(n + 1)]
        dp[0][0] = 0

        for i in range(1, n + 1):
            for j in range(501):
                # If the paid painter is not occupied, use the paid painter
                if j >= time[i - 1]:
                    dp[i][j] = min(dp[i][j], dp[i - 1][j - time[i - 1]] + cost[i - 1])
                # If the paid painter is occupied, use the free painter
                dp[i][j] = min(dp[i][j], dp[i - 1][j - 1])

        return min(dp[n])