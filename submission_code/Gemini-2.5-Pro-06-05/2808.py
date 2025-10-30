import math
from typing import List

class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        """
        Solves the paint walls problem using dynamic programming, framed as a 0/1 knapsack problem.
        """
        n = len(cost)
        
        # dp[j] represents the minimum cost to achieve a "coverage" of at least j walls.
        # When we pay for wall i, it gets painted (1 wall) and the free painter can paint
        # time[i] walls in parallel. So, the total coverage from this action is 1 + time[i].
        # We need to find the minimum cost to achieve a total coverage of at least n.
        
        # dp table size is n+1, for coverages 0 to n.
        # Initialize with a large value representing infinity.
        dp = [math.inf] * (n + 1)
        
        # Base case: It costs 0 to cover 0 walls.
        dp[0] = 0
        
        # Iterate through each wall, treating it as an item in a knapsack problem.
        for i in range(n):
            current_cost = cost[i]
            # The number of walls this single paid job covers.
            coverage = 1 + time[i]
            
            # Update the dp table. Iterate backwards to ensure each wall is used at most once.
            for j in range(n, 0, -1):
                # To achieve a coverage of at least `j`, we can potentially use wall `i`.
                # If we use wall `i`, we need to have previously covered at least `j - coverage` walls.
                # The index for the previous state is `max(0, j - coverage)`.
                # We use `max(0, ...)` because if `j - coverage` is negative, it means wall `i` alone
                # provides enough coverage. In this case, the previous state needed is for 0 coverage,
                # which costs `dp[0] = 0`.
                prev_coverage_idx = max(0, j - coverage)
                
                # The new cost to achieve coverage `j` is the cost to achieve the
                # previous coverage plus the cost of the current wall.
                # We take the minimum with the existing dp[j] (which represents
                # not using the current wall `i` to achieve state `j`).
                dp[j] = min(dp[j], dp[prev_coverage_idx] + current_cost)
                
        # The answer is the minimum cost to cover at least n walls.
        return dp[n]