import math
# Assuming List is available from context (e.g., from typing import List)
# as per the problem's starter code format.

class Solution:
  def paintWalls(self, cost: List[int], time: List[int]) -> int:
    num_walls = len(cost)

    # dp[j] will store the minimum cost to achieve a "coverage" of j.
    # "Coverage" means: (number of walls painted by the paid painter) + 
    #                   (number of walls that can be painted by the free painter 
    #                    during the time the paid painter is busy).
    # We need to achieve a total coverage of at least num_walls.
    # The size of dp array needs to be num_walls + 1 to store dp[0] through dp[num_walls].
    
    # Initialize dp[0] = 0 (0 cost to cover 0 walls/slots).
    # Initialize dp[j] = infinity for j > 0, as initially we don't know how to cover them.
    # math.inf works well for representing infinity.
    dp = [math.inf] * (num_walls + 1)
    dp[0] = 0

    # Iterate through each wall. Each wall i (specified by cost[i] and time[i])
    # can be thought of as an item in a 0/1 knapsack-like problem.
    for i in range(num_walls):
      current_wall_cost = cost[i]
      
      # If the paid painter paints wall i:
      # 1. It costs `cost[i]`.
      # 2. This wall (1 wall) gets painted by the paid painter.
      # 3. The paid painter is occupied for `time[i]` units.
      # 4. During these `time[i]` units, the free painter can paint `time[i]` other walls.
      # So, choosing to paint wall `i` with the paid painter contributes `1 + time[i]`
      # to the total "coverage" or "slots filled".
      coverage_provided_by_this_wall = 1 + time[i]

      # Update the dp table. We iterate j downwards to ensure that for each dp state,
      # the decision for wall `i` is based on states derived from *previous* walls (0 to i-1),
      # effectively meaning each wall is considered at most once (0/1 knapsack property).
      # j represents the target coverage we are trying to achieve.
      for j in range(num_walls, 0, -1): # j iterates from num_walls down to 1
        
        # We want to calculate dp[j]. We have two choices for wall `i`:
        # 1. Don't use paid painter for wall `i` to contribute to this coverage `j`.
        #    In this case, dp[j] remains its value computed using walls 0 to `i-1`.
        #    This is implicitly handled when we take the minimum.
        # 2. Use paid painter for wall `i`.
        #    The cost incurred for this wall is `current_wall_cost`.
        #    This wall provides `coverage_provided_by_this_wall`.
        #    So, we must have achieved `j - coverage_provided_by_this_wall` coverage
        #    from walls 0 to `i-1` (or other walls already processed in the outer loop).
        #    If `j - coverage_provided_by_this_wall` is negative, it means this wall alone
        #    provides enough (or more than enough) coverage for `j`.
        #    The cost from "previous items" to make up the remaining coverage would be `dp[0] = 0`.
        #    This is handled by `dp[max(0, j - coverage_provided_by_this_wall)]`.
        
        # Cost if we choose to paint wall `i` with the paid painter:
        # It's the sum of `current_wall_cost` and the cost to achieve the remaining needed coverage.
        cost_to_achieve_remaining_coverage = dp[max(0, j - coverage_provided_by_this_wall)]
        
        # If cost_to_achieve_remaining_coverage is math.inf, it means that state was not reachable.
        # Adding current_wall_cost to math.inf still results in math.inf.
        cost_if_using_this_wall_for_paid_painter = cost_to_achieve_remaining_coverage + current_wall_cost
        
        dp[j] = min(dp[j], cost_if_using_this_wall_for_paid_painter)

    # After considering all walls, dp[num_walls] will hold the minimum cost
    # to achieve a coverage of at least num_walls.
    return dp[num_walls]