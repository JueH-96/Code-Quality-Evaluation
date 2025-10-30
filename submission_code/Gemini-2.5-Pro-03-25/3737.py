import math
from typing import List

class Solution:
  """
  Solves the minimum cost painting problem with adjacent and equidistant constraints using dynamic programming.
  The approach uses dynamic programming processing houses in pairs from outwards to inwards.
  Houses are indexed from 0 to n-1. The pairs considered are (i, n-1-i) for i from 0 to n/2 - 1.
  The dynamic programming state dp[i][c1][c2] would represent the minimum cost to paint houses 0..i and n-1-i..n-1,
  such that house i has color c1 and house n-1-i has color c2.
  To optimize space, we only keep track of the DP states for the previous step (i-1).

  Space complexity is O(1) because the DP state space size is constant (3x3 = 9).
  Time complexity is O(N) because we iterate through N/2 pairs, and for each pair, we perform a constant number of operations
  (iterating through current state colors 3x3=9 and previous state colors 3x3=9, total 81 operations per pair).
  """
  def minCost(self, n: int, cost: List[List[int]]) -> int:
    """
    Calculates the minimum cost to paint n houses satisfying the given conditions.

    Args:
        n: An even integer representing the number of houses.
        cost: A 2D list where cost[i][j] is the cost of painting house i with color j (0, 1, or 2).

    Returns:
        The minimum cost to paint the houses beautifully. Returns infinity if no solution exists,
        but problem constraints imply a solution always exists.
    """

    # dp_prev[c1][c2] stores the minimum cost for houses processed up to the pair (i-1, n-i),
    # where house i-1 was painted with color c1 and house n-i was painted with color c2.
    # Initialized to represent the base case pair (0, n-1).
    dp_prev = [[math.inf] * 3 for _ in range(3)]
    
    # Base case: Consider the first pair of houses (0, n-1).
    # Initialize dp_prev with the costs for painting this pair.
    for c0 in range(3):  # Color for house 0
        for c_n_minus_1 in range(3):  # Color for house n-1
            # Equidistant constraint: house 0 and house n-1 must have different colors.
            if c0 != c_n_minus_1:
                dp_prev[c0][c_n_minus_1] = cost[0][c0] + cost[n - 1][c_n_minus_1]

    # DP iterations: Process pairs (i, n-1-i) for i from 1 up to n/2 - 1.
    # We are calculating the states for pair i based on states from pair i-1.
    for i in range(1, n // 2):
        # Initialize DP table for the current step (pair i, n-1-i).
        dp_curr = [[math.inf] * 3 for _ in range(3)]
        
        # Iterate through all possible color combinations (ci, c_opp) for the current pair (i, n-1-i).
        # ci: color of house i. c_opp: color of house n-1-i.
        for ci in range(3):
            for c_opp in range(3):
                # Equidistant constraint for pair (i, n-1-i): house i and house n-1-i must have different colors.
                if ci == c_opp:
                    continue  # Skip if colors are the same, this state configuration is invalid.
                
                # Cost of painting the current pair of houses (i, n-1-i).
                current_pair_cost = cost[i][ci] + cost[n - 1 - i][c_opp]
                
                min_prev_total_cost = math.inf
                
                # Iterate over all possible color combinations (c_prev_i, c_prev_opp) for the previous pair (i-1, n-i).
                # c_prev_i: color of house i-1. c_prev_opp: color of house n-i.
                for c_prev_i in range(3):
                    for c_prev_opp in range(3):
                        
                        # Retrieve cost from the previous state stored in dp_prev.
                        prev_cost = dp_prev[c_prev_i][c_prev_opp]
                        if prev_cost == math.inf:
                            continue # Skip if the previous state was invalid or unreachable.

                        # Check adjacent constraints:
                        # House i (color ci) must differ from house i-1 (color c_prev_i).
                        # House n-1-i (color c_opp) must differ from house n-i (color c_prev_opp).
                        if c_prev_i != ci and c_prev_opp != c_opp:
                            # If constraints are met, this path is valid. Update the minimum cost found from valid previous states.
                            min_prev_total_cost = min(min_prev_total_cost, prev_cost)
                
                # If a valid path from previous states exists (min_prev_total_cost is finite).
                if min_prev_total_cost != math.inf:
                     # Update the DP table for the current state (ci, c_opp).
                     # The cost is the sum of the current pair's cost and the minimum cost to reach this state configuration.
                     dp_curr[ci][c_opp] = current_pair_cost + min_prev_total_cost

        # Update dp_prev for the next iteration. The current DP table becomes the previous one.
        dp_prev = dp_curr 

    # After the loop finishes, dp_prev holds the minimum costs for painting all n houses.
    # The indices in dp_prev[c1][c2] correspond to the colors of the innermost pair (n/2 - 1, n/2).
    # c1 is the color of house n/2 - 1, c2 is the color of house n/2.
    min_total_cost = math.inf
    for c1 in range(3):
        for c2 in range(3):
             # Find the minimum value in the final DP table.
             # States where c1 == c2 will have infinity cost due to the equidistant constraint check
             # applied at the last step (i = n/2 - 1) for the pair (n/2 - 1, n/2).
             min_total_cost = min(min_total_cost, dp_prev[c1][c2])
            
    # If min_total_cost remained infinity, it would mean no solution exists.
    # Given the problem constraints and the availability of 3 colors, a solution should always be possible.
    # Return the minimum cost found. Check if infinity was returned, though unlikely needed.
    return min_total_cost if min_total_cost != math.inf else -1 # Problem implies cost >= 0, so returning inf or -1 is more of an error indicator.