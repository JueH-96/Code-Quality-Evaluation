from typing import List
import math

class Solution:
    def minCost(self, n: int, cost: List[List[int]]) -> int:
        # This problem can be solved using Dynamic Programming.
        # We consider painting the houses in pairs from the ends inwards:
        # (0, n-1), (1, n-2), ..., (n/2 - 1, n/2).
        # Let dp[i][c1][c2] be the minimum cost to paint the houses from index 0 to i
        # and from index n-1-i to n-1, such that house i has color c1 and house n-1-i has color c2.
        # The colors c1 and c2 must be different according to the equidistant constraint.
        # The transition from dp[i-1] to dp[i] involves considering the costs of painting
        # the current pair (i, n-1-i) and ensuring adjacent houses (i-1 and i, n-i and n-1-i)
        # have different colors.
        # Since the DP state for step i only depends on step i-1, we can optimize space
        # by using only two 3x3 tables: prev_dp and curr_dp.

        # prev_dp[prev_c1][prev_c2] stores min cost to paint houses up to pair (i-1, n-i),
        # where house i-1 has color prev_c1 and house n-i has color prev_c2.
        # curr_dp[curr_c1][curr_c2] stores min cost to paint houses up to pair (i, n-1-i),
        # where house i has color curr_c1 and house n-1-i has color curr_c2.
        # The "houses up to pair (k, n-1-k)" means all houses from index 0 to k AND
        # all houses from index n-1-k to n-1, painted according to the rules.

        # Initialize prev_dp for the first pair (0, n-1)
        # prev_c1 is color for house 0, prev_c2 is color for house n-1.
        prev_dp = [[float('inf')] * 3 for _ in range(3)]
        house0_idx = 0
        housen_1_idx = n - 1

        for c0 in range(3): # color for house 0
            for cn_1 in range(3): # color for house n-1
                # Equidistant constraint for the first pair (0, n-1)
                if c0 != cn_1:
                    prev_dp[c0][cn_1] = cost[house0_idx][c0] + cost[housen_1_idx][cn_1]

        # Iterate through pairs from i = 1 up to n/2 - 1
        # At iteration i, we consider pair (i, n-1-i)
        # and depend on the costs calculated for pair (i-1, n-i) stored in prev_dp
        for i in range(1, n // 2):
            curr_dp = [[float('inf')] * 3 for _ in range(3)]
            house_i_idx = i # Current house index 1 (house i)
            house_n_1_i_idx = n - 1 - i # Current house index 2 (house n-1-i)

            # Choose colors (curr_c1, curr_c2) for the current pair (house_i_idx, house_n_1_i_idx)
            for curr_c1 in range(3): # color for house i
                for curr_c2 in range(3): # color for house n-1-i
                    # Equidistant constraint for the current pair (i, n-1-i)
                    if curr_c1 == curr_c2:
                        continue

                    current_pair_cost = cost[house_i_idx][curr_c1] + cost[house_n_1_i_idx][curr_c2]
                    min_prev_cost = float('inf')

                    # Iterate through possible colors (prev_c1, prev_c2) for the previous pair (i-1, n-i)
                    # These costs are stored in prev_dp
                    # prev_c1 is color for house i-1, prev_c2 is color for house n-i

                    for prev_c1 in range(3): # color for house i-1
                        for prev_c2 in range(3): # color for house n-i
                            # Adjacent constraints:
                            # house i-1 (prev_c1) != house i (curr_c1)
                            # house n-i (prev_c2) != house n-1-i (curr_c2)
                            # Also implicitly requires prev_c1 != prev_c2 (equidistant for previous pair),
                            # which is handled by checking prev_dp[prev_c1][prev_c2] != float('inf').

                            if prev_c1 != curr_c1 and prev_c2 != curr_c2:
                                # Consider the cost of the previous valid state (prev_c1, prev_c2) for pair (i-1, n-i)
                                if prev_dp[prev_c1][prev_c2] != float('inf'):
                                    min_prev_cost = min(min_prev_cost, prev_dp[prev_c1][prev_c2])

                    if min_prev_cost != float('inf'):
                        curr_dp[curr_c1][curr_c2] = current_pair_cost + min_prev_cost

            # Update prev_dp for the next iteration
            prev_dp = curr_dp

        # The final answer is the minimum cost in the last calculated DP table (prev_dp)
        # over all valid pairs of colors (c1, c2) for the last pair of houses (n/2-1, n/2)
        min_total_cost = float('inf')
        # The last prev_dp corresponds to pair (n/2 - 1, n/2).
        # The indices in the last prev_dp are colors for house n/2 - 1 and house n/2.
        # Let's use generic variable names c1, c2 for the colors of the last pair
        # which are houses n/2 - 1 and n/2.
        for c1 in range(3): # color for house n/2 - 1
            for c2 in range(3): # color for house n/2
                 # The DP transition already ensured c1 != c2 for finite values,
                 # but adding the check here makes the final result extraction clear.
                 if c1 != c2:
                     min_total_cost = min(min_total_cost, prev_dp[c1][c2])

        return min_total_cost