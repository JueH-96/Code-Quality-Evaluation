from typing import List

class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        # We want to choose a subset of walls for which the paid painter paints.
        # The free painter can paint walls for free, but only when the paid painter is busy.
        # If we let S be the set of walls painted by the paid painter,
        # then the paid painter is busy for sum(time[i] for i in S) units.
        # The free painter paints each wall in 1 unit time.
        # Since there are n - |S| walls needing the free painter,
        # we require that sum(time[i] for i in S) >= n - |S|.
        #
        # Rearranging the inequality, we have:
        #     |S| + sum(time[i] for i in S) >= n.
        #
        # Notice that each wall i painted by the paid painter contributes a "capacity"
        # of (time[i] + 1) toward covering the n walls.
        # Our task is to select a set S with a minimal total cost,
        # such that the total capacity is at least n.
        #
        # We can solve this via a knapSack-style DP:
        # Let dp[v] be the minimal cost needed to achieve a total capacity v,
        # where we clamp v to at most n (since that’s all we need).
        
        INF = float('inf')
        dp = [INF] * (n + 1)
        dp[0] = 0
        
        for c, t in zip(cost, time):
            contribution = t + 1  # This wall contributes (time + 1) capacity.
            # We update in reverse order to avoid using the same wall twice.
            for cap in range(n, -1, -1):
                if dp[cap] != INF:
                    new_cap = cap + contribution
                    if new_cap > n:
                        new_cap = n
                    dp[new_cap] = min(dp[new_cap], dp[cap] + c)
        
        return dp[n]