from typing import List
import heapq

class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        jobs = sorted(zip(cost, time))
        dp = [0] + [float('inf')] * n
        for c, t in jobs:
            for i in range(n, 0, -1):
                dp[i] = min(dp[i], dp[max(i - t - 1, 0)] + c)
        return dp[-1]