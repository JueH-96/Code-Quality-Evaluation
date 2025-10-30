from typing import List
import math

class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        if n == 0:
            return 0
        
        # Calculate the maximum s we need to consider
        max_time_plus_one = max(t + 1 for t in time)
        s_max = n + max_time_plus_one
        # To be safe, set s_max to n + max(time[i] + 1)
        s_max = min(s_max, n + 501)  # Since time[i] <= 500
        
        # Initialize dp array
        dp = [math.inf] * (s_max + 1)
        dp[0] = 0
        
        # Iterate through each wall
        for i in range(n):
            # Iterate s from high to low to prevent using the same wall multiple times
            for s in range(s_max, -1, -1):
                if dp[s] != math.inf:
                    s_new = s + time[i] + 1
                    if s_new > s_max:
                        s_new = s_max
                    dp[s_new] = min(dp[s_new], dp[s] + cost[i])
        
        # Find the minimum cost where s >= n
        min_cost = min(dp[s] for s in range(n, s_max + 1))
        
        return min_cost