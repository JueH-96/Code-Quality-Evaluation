import math
from typing import List

class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        
        # Calculate the maximum possible time taken by a single paid painter.
        # This is used to determine the necessary size of our DP array.
        max_time_val = 0
        if n > 0: # Ensures time list is not empty before calling max()
            max_time_val = max(time)
        
        # max_j_needed represents the maximum 'effective walls' index we might need to consider.
        # If we have painted 'k' walls effectively, where k < n, and we choose
        # to paint one more wall 'i' with paid painter, it yields '1 + time[i]'
        # effective walls. The new total is k + (1 + time[i]).
        # The largest relevant k is (n - 1).
        # The largest (1 + time[i]) is (1 + max_time_val).
        # So, the maximum index in dp array that could be a result of a direct
        # push from a state less than n is (n - 1) + (1 + max_time_val) = n + max_time_val.
        max_j_needed = n + max_time_val
        
        # dp[j] will store the minimum cost to achieve 'j' effective walls.
        # Initialize with infinity, except dp[0] which is 0 cost for 0 effective walls.
        dp = [float('inf')] * (max_j_needed + 1)
        dp[0] = 0
        
        # Iterate through each wall.
        # This is a 0/1 knapsack-like problem, so each wall is considered once.
        for i in range(n):
            current_cost = cost[i]
            # Each wall 'i' painted by paid painter provides 1 actual wall painted
            # and 'time[i]' units of time during which the free painter can work.
            # So, it contributes '1 + time[i]' to the total effective walls.
            effective_walls_gained = 1 + time[i]
            
            # Iterate 'j' downwards to ensure that each wall is used at most once in the
            # calculation of the current DP state.
            # We iterate down from max_j_needed to effective_walls_gained.
            # If j - effective_walls_gained < 0, then we cannot form 'j' effective walls
            # by adding 'effective_walls_gained' to a previous state.
            for j in range(max_j_needed, effective_walls_gained - 1, -1):
                # Check if the previous state (j - effective_walls_gained) was reachable.
                if dp[j - effective_walls_gained] != float('inf'):
                    # Update dp[j] with the minimum of its current value and
                    # the cost of reaching j - effective_walls_gained plus the current wall's cost.
                    dp[j] = min(dp[j], dp[j - effective_walls_gained] + current_cost)
        
        # After processing all walls, find the minimum cost among all states that
        # have painted at least 'n' effective walls.
        min_total_cost = float('inf')
        for j in range(n, max_j_needed + 1):
            min_total_cost = min(min_total_cost, dp[j])
            
        return min_total_cost