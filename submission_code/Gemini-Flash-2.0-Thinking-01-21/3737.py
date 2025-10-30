import math
from typing import List

class Solution:
    def minCost(self, n: int, cost: List[List[int]]) -> int:
        # dp[c1][c2] will store the minimum cost to paint the first i pairs
        # (houses 0..i and n-1-i..n-1) such that house i has color c1 and
        # house n-1-i has color c2. c1, c2 are 0, 1, or 2 (representing colors 1, 2, 3).

        # Initialize dp for the first pair (houses 0 and n-1)
        # This represents dp[0][c1][c2]
        prev_dp = [[math.inf] * 3 for _ in range(3)]
        for c1 in range(3):
            for c2 in range(3):
                # Equidistant constraint for the first pair (0, n-1): house 0 color != house n-1 color
                if c1 != c2:
                    prev_dp[c1][c2] = cost[0][c1] + cost[n - 1][c2]
                # else: prev_dp[c1][c2] remains math.inf

        # Iterate through the remaining pairs (i, n-1-i) from i=1 to n/2 - 1
        # The loop processes pairs (1, n-2), (2, n-3), ..., (n/2 - 1, n/2).
        # The total number of pairs is n/2.
        for i in range(1, n // 2):
            curr_dp = [[math.inf] * 3 for _ in range(3)]

            # Collect valid previous costs and their corresponding colors for pair (i-1, n-i)
            # These are states from the previous DP step (i-1) where the equidistant constraint (pc1 != pc2) was satisfied.
            prev_costs = []
            for pc1 in range(3):
                for pc2 in range(3):
                    # Only consider valid previous states (where cost is not infinity)
                    # prev_dp[pc1][pc2] != math.inf implicitly means pc1 != pc2
                    if prev_dp[pc1][pc2] != math.inf:
                         prev_costs.append((prev_dp[pc1][pc2], (pc1, pc2)))

            # Sort previous costs to find the minimums efficiently
            # This list will have at most 3*2 = 6 elements (valid previous states). Sorting is O(1).
            # Sorting by cost_val primarily.
            prev_costs.sort()

            # Iterate through all possible color combinations for the current pair (i, n-1-i)
            for c1 in range(3):
                for c2 in range(3):
                    # Current pair equidistant constraint: house i color != house n-1-i color
                    if c1 != c2:
                        # Find the minimum previous cost (from painting pair i-1)
                        # that satisfies adjacent constraints with the current pair i.
                        # The previous colors (pc1, pc2) must not be equal to the current colors (c1, c2) respectively.
                        # pc1 != c1 (Adjacent constraint: color of house i-1 != color of house i)
                        # pc2 != c2 (Adjacent constraint: color of house n-i != color of house n-1-i)
                        min_prev_cost = math.inf
                        for cost_val, (pc1, pc2) in prev_costs:
                            if pc1 != c1 and pc2 != c2:
                                min_prev_cost = cost_val
                                break # Found the minimum valid cost

                        # If a valid previous state was found (min_prev_cost is not inf)
                        if min_prev_cost != math.inf:
                            curr_dp[c1][c2] = cost[i][c1] + cost[n - 1 - i][c2] + min_prev_cost
                        # else: curr_dp[c1][c2] remains math.inf

            # Update prev_dp for the next iteration
            prev_dp = curr_dp

        # After processing all pairs (from 0 to n/2 - 1), the minimum cost is the minimum value
        # in the final prev_dp table. This table stores costs for the last pair (n/2 - 1, n/2).
        # We need the overall minimum cost among all valid final states (where c1 != c2, which is guaranteed if cost is not inf).
        min_total_cost = math.inf
        for r in range(3):
            min_total_cost = min(min_total_cost, min(prev_dp[r]))

        return min_total_cost