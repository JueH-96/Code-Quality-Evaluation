from typing import List

class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        for i in range(n):
            time_plus_1 = time[i] + 1
            current_cost = cost[i]
            for j in range(n, -1, -1):
                if dp[j] != float('inf'):
                    new_j = j + time_plus_1
                    if new_j > n:
                        new_j = n
                    if dp[new_j] > dp[j] + current_cost:
                        dp[new_j] = dp[j] + current_cost
        return dp[n]