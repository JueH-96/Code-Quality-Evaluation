import math
from typing import List

class Solution:
  """
    Solves the paint walls problem using dynamic programming.
    The problem asks for the minimum cost to paint n walls given costs and times for a paid painter,
    and a free painter that works only when the paid painter is busy.

    Let P be the set of walls painted by the paid painter. The cost is sum(cost[i] for i in P).
    The total time the paid painter works is T = sum(time[i] for i in P).
    The number of walls painted by the paid painter is p = |P|.
    The remaining n - p walls must be painted by the free painter.
    The free painter takes 1 unit of time per wall. To paint n - p walls, the free painter needs n - p time units.
    The free painter can only work when the paid painter is working. The total time available for the free painter is T.
    So, we must have T >= n - p. This is equivalent to T + p >= n.

    Let v_i = time[i] + 1. This represents the "effective walls covered" by painting wall i with the paid painter. 
    It covers wall i itself (contributes 1 to p) and provides time[i] time units for the free painter (contributes time[i] to T).
    The condition becomes sum(v_i for i in P) >= n.
    We want to find a subset P that minimizes sum(cost[i] for i in P) subject to sum(v_i for i in P) >= n.

    This is a variation of the 0/1 knapsack problem: find the minimum cost to achieve a total value (effective walls covered) of at least n.
    We can use dynamic programming. Let dp[j] be the minimum cost to achieve a total value (effective walls covered) of *at least* j.
    The size of the DP table needed is n + 1 (indices from 0 to n).

    The state transition is derived by considering each wall i: either we don't include it in the set P (painted by paid painter), or we do.
    If we don't include wall i, the minimum cost to achieve at least j effective walls remains the same as before considering wall i.
    If we include wall i (costing cost[i] and providing v_i effective walls), we need to have achieved at least max(0, j - v_i) effective walls using the previous walls. The cost would be dp[max(0, j - v_i)] + cost[i].
    So, dp[j] = min(dp[j]_without_wall_i, dp[max(0, j - v_i)]_without_wall_i + cost[i]).

    To implement this using a 1D DP array, we iterate through the walls and update the DP table. Iterating j from n down to 1 ensures that when we calculate dp[j] using dp[prev_state_idx], the value dp[prev_state_idx] corresponds to the state *before* considering the current wall i.
  """
  def paintWalls(self, cost: List[int], time: List[int]) -> int:
    """
    Calculates the minimum cost to paint n walls.

    Args:
      cost: A list of integers where cost[i] is the cost to paint wall i with the paid painter.
      time: A list of integers where time[i] is the time taken by the paid painter for wall i.

    Returns:
      The minimum cost required to paint all n walls.
    """
    n = len(cost)
    
    # Initialize DP array of size n+1. dp[j] stores min cost for at least j effective walls.
    # Initialize with infinity, representing unreachable states.
    dp = [math.inf] * (n + 1)
    # Base case: 0 cost to achieve at least 0 effective walls.
    dp[0] = 0 

    # Iterate through each wall
    for i in range(n):
        c = cost[i]
        t = time[i]
        # The effective walls covered if this wall i is painted by the paid painter:
        # v = p_contribution + T_contribution = 1 + time[i]
        v = 1 + t 

        # Update dp table states from right to left (from n down to 1)
        # This ensures that we correctly use the states from the previous iteration (without wall i).
        for j in range(n, 0, -1): 
            # Calculate the index of the state needed from the previous iteration (before considering wall i)
            # If we choose wall i, we need to have achieved at least j - v effective walls previously.
            # We use max(0, ...) because the effective walls count cannot be negative.
            prev_state_idx = max(0, j - v)
            
            # If the previous state dp[prev_state_idx] is reachable (not infinity cost)
            if dp[prev_state_idx] != math.inf:
                # Update dp[j]: it's the minimum of its current value and the cost of including wall i.
                # The cost of including wall i is cost[i] plus the minimum cost to reach the required previous state.
                dp[j] = min(dp[j], dp[prev_state_idx] + c)

    # The final answer is dp[n], which stores the minimum cost to achieve at least n effective walls.
    # This ensures that all n walls can be painted (either by paid or free painter within the paid painter's total working time).
    # The problem constraints and logic imply that achieving at least n effective walls is always possible.
    # For instance, painting all walls with the paid painter yields n + sum(time) effective walls, which is >= n since time[i] >= 1.
    return dp[n]