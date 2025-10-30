from typing import List
import math

class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        if n == 0:
            return 0
        
        # Calculate the maximum possible total time (sum of 1 + time[i] for all walls)
        max_t = sum(1 + time[i] for i in range(n))
        
        # Initialize DP array with infinity, except dp[0] = 0
        dp = [math.inf] * (max_t + 1)
        dp[0] = 0
        
        for i in range(n):
            weight = 1 + time[i]
            cost_i = cost[i]
            # Iterate backwards to prevent overwriting values we need to reuse
            for t in range(max_t, -1, -1):
                if dp[t] != math.inf:
                    new_t = t + weight
                    if new_t > max_t:
                        new_t = max_t
                    if dp[new_t] > dp[t] + cost_i:
                        dp[new_t] = dp[t] + cost_i
        
        # Find the minimum cost for t >= n
        min_cost = min(dp[n:max_t + 1])
        return min_cost