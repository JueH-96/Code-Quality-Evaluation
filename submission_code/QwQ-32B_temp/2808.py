from typing import List

class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        INF = float('inf')
        dp = [INF] * (n + 1)
        dp[0] = 0  # Base case: 0 cost to achieve 0 total
        
        for c, t in zip(cost, time):
            contribution = t + 1
            # Iterate backwards to avoid reusing the same wall multiple times
            for j in range(n, -1, -1):
                if dp[j] != INF:
                    new_j = j + contribution
                    if new_j > n:
                        new_j = n
                    if dp[new_j] > dp[j] + c:
                        dp[new_j] = dp[j] + c
        
        return dp[n]