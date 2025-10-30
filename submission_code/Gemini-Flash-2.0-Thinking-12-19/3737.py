from typing import List
import sys

class Solution:
    def minCost(self, n: int, cost: List[List[int]]) -> int:
        # Use a large value for infinity
        INF = float('inf')

        # dp_prev and dp_curr are 3x3 tables
        # dp[c1][c2] stores the minimum cost to paint the first i pairs
        # (houses 0..i-1 and n-i..n-1) such that house i-1 has color c1
        # and house n-i has color c2.
        # Colors are indexed 0, 1, 2

        # Base case: i = 1 (first pair of houses: 0 and n-1)
        # c1 is color for house 0, c2 is color for house n-1
        dp_prev = [[INF] * 3 for _ in range(3)]
        for c1 in range(3):
            for c2 in range(3):
                # Houses 0 and n-1 must have different colors
                if c1 != c2:
                    dp_prev[c1][c2] = cost[0][c1] + cost[n - 1][c2]

        # DP transition: i from 2 to n/2
        # i represents the number of pairs considered.
        # At step i, we consider houses i-1 and n-i.
        # The previous step i-1 considered houses i-2 and n-i+1.
        for i in range(2, n // 2 + 1):
            dp_curr = [[INF] * 3 for _ in range(3)]
            
            # c1 is color for house i-1
            # c2 is color for house n-i
            for c1 in range(3):
                for c2 in range(3):
                    # Houses i-1 and n-i must have different colors
                    if c1 == c2:
                        continue

                    min_prev_cost = INF

                    # pc1 is color for house i-2
                    # pc2 is color for house n-i+1
                    for pc1 in range(3):
                        # House i-1 (color c1) must be different from house i-2 (color pc1)
                        if pc1 == c1:
                            continue

                        for pc2 in range(3):
                            # House n-i (color c2) must be different from house n-i+1 (color pc2)
                            if pc2 == c2:
                                continue

                            # Houses i-2 (color pc1) and n-i+1 (color pc2) must have different colors.
                            if pc1 == pc2:
                                continue

                            # Consider the minimum cost from the previous state (i-1), dp_prev[pc1][pc2]
                            # Ensure the previous state is valid (not INF).
                            if dp_prev[pc1][pc2] != INF:
                                min_prev_cost = min(min_prev_cost, dp_prev[pc1][pc2])

                    # If a valid sequence of previous paintings exists leading to this state
                    # and we found a valid previous state (min_prev_cost is not INF)
                    if min_prev_cost != INF:
                        # Add the cost of painting houses i-1 and n-i
                        dp_curr[c1][c2] = cost[i - 1][c1] + cost[n - i][c2] + min_prev_cost

            # Move to the next iteration
            dp_prev = dp_curr

        # The result is the minimum value in the final dp_prev table (for i = n/2)
        # This table contains the minimum cost for painting all n houses,
        # with house n/2 - 1 colored c1 and house n/2 colored c2.
        # We need the minimum cost over all valid color pairs (c1, c2) for the last pair.
        min_total_cost = INF
        for c1 in range(3):
            for c2 in range(3):
                 # dp_prev[c1][c2] is INF if c1 == c2 or if no valid path leads here
                 min_total_cost = min(min_total_cost, dp_prev[c1][c2])

        return min_total_cost