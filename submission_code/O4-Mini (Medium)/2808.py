from typing import List

class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        # We need the sum of (time[i] + 1) over chosen paid-walls >= n
        # i.e., choose a subset S to minimize sum(cost[i]) s.t. sum(b_i) >= n,
        # where b_i = time[i] + 1.  This is a 0/1 knapsack-style DP.
        
        INF = 10**18
        target = n
        # dp[j] = minimum cost to achieve total "capacity" >= j
        dp = [INF] * (target + 1)
        dp[0] = 0
        
        for c, t in zip(cost, time):
            b = t + 1
            # Traverse backwards to avoid reusing the same item
            for j in range(target, -1, -1):
                prev = dp[j]
                need = j - b
                if need < 0:
                    need = 0
                cand = dp[need] + c
                if cand < prev:
                    dp[j] = cand
        
        return dp[target]