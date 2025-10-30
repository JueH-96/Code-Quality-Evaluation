import math
from typing import List

class Solution:
  def minCost(self, n: int, cost: List[List[int]]) -> int:
    # dp_prev[c1][c2] stores the minimum cost for painting pairs up to the "previous" ones,
    # where c1 was the color of the left house and c2 the color of the right house of that pair.
    # Initialize for the outermost pair (house 0, house n-1).
    # c_L: color for house 0. c_R: color for house n-1.
    dp_prev = [[math.inf] * 3 for _ in range(3)]

    for c_L in range(3):
        for c_R in range(3):
            if c_L == c_R:  # Equidistant constraint: color(0) != color(n-1)
                continue
            dp_prev[c_L][c_R] = cost[0][c_L] + cost[n-1][c_R]

    # Iterate for subsequent pairs, moving inwards.
    # k ranges from 1 to n//2 - 1.
    # Current pair: (house k, house n-1-k).
    # Previous pair was: (house k-1, house n-k).
    for k in range(1, n // 2):
        dp_curr = [[math.inf] * 3 for _ in range(3)]
        
        # curr_c_L: color for house k.
        # curr_c_R: color for house n-1-k.
        for curr_c_L in range(3):
            for curr_c_R in range(3):
                if curr_c_L == curr_c_R:  # Equidistant constraint for current pair k
                    continue

                current_pair_actual_cost = cost[k][curr_c_L] + cost[n-1-k][curr_c_R]
                
                min_cost_from_prev_pairs = math.inf

                # prev_c_L: color for house k-1.
                # prev_c_R: color for house n-k.
                for prev_c_L in range(3):
                    for prev_c_R in range(3):
                        # Adjacency constraints:
                        if curr_c_L == prev_c_L:  # color(k) != color(k-1)
                            continue
                        if curr_c_R == prev_c_R:  # color(n-1-k) != color(n-k)
                            continue
                        
                        # dp_prev[prev_c_L][prev_c_R] would be math.inf if prev_c_L == prev_c_R (equidistant constraint for pair k-1).
                        # min() handles math.inf correctly.
                        min_cost_from_prev_pairs = min(min_cost_from_prev_pairs, dp_prev[prev_c_L][prev_c_R])
                
                # If min_cost_from_prev_pairs is math.inf, it means no valid way to paint previous pairs
                # to satisfy constraints leading to (curr_c_L, curr_c_R) for the current pair.
                # In this case, dp_curr[curr_c_L][curr_c_R] remains math.inf.
                if min_cost_from_prev_pairs != math.inf:
                    dp_curr[curr_c_L][curr_c_R] = current_pair_actual_cost + min_cost_from_prev_pairs
        
        dp_prev = dp_curr  # Current state becomes previous state for the next iteration.

    # After the loop, dp_prev holds costs for the innermost pair (n/2-1, n/2).
    # The minimum value in this table is the overall minimum cost.
    min_total_cost = math.inf
    for c_L in range(3):
        for c_R in range(3):
            # dp_prev[c_L][c_R] is math.inf if c_L == c_R due to the equidistant constraint.
            min_total_cost = min(min_total_cost, dp_prev[c_L][c_R])
            
    # The problem implies a solution always exists with finite cost.
    return int(min_total_cost)