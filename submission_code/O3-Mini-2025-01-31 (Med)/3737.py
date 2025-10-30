from typing import List
import math

class Solution:
    def minCost(self, n: int, cost: List[List[int]]) -> int:
        # We group houses into pairs:
        # For i from 0 to half-1, pair i contains house i (left) and house n-1-i (right).
        # For each pair, we choose a color for left and a color for right under the symmetry constraint: left != right.
        # Additionally, adjacent houses in the row (which occur as:
        #   left_i (for i=0...half-2) adjacent to left_{i+1},
        #   and, in the second half, the houses appear in reverse order so that for i=1...half-1, right_i is adjacent to right_{i-1})
        # must be painted different colors.
        # Thus, for pairs with indices i (>=1), if we choose (L, R) for pair i and (L_prev, R_prev) for pair i-1, we must have:
        #    L != L_prev   and   R != R_prev.
        #
        # We run a DP over these pairs.
        
        half = n // 2
        
        # Precompute all valid color choices for a pair: (left_color, right_color) with left != right.
        valid_pairs = [(l, r) for l in range(3) for r in range(3) if l != r]  # 6 states
        
        # dp[i][s] = minimum cost covering pairs 0 to i, ending with pair i using state s (which is a valid (L,R) assignment)
        dp = [[math.inf] * len(valid_pairs) for _ in range(half)]
        
        # Base: pair 0 corresponds to houses[0] and houses[n-1]
        for idx, (l, r) in enumerate(valid_pairs):
            dp[0][idx] = cost[0][l] + cost[n-1][r]
        
        # Transition: for pair i (i from 1 to half-1), choose state (l, r)
        for i in range(1, half):
            # Cost for current pair i is:
            # cost[i][l] for house i (left) and cost[n-1-i][r] for right house.
            for idx, (l, r) in enumerate(valid_pairs):
                cur_cost = cost[i][l] + cost[n-1-i][r]
                best_prev = math.inf
                # Examine transitions from previous pair's state (l_prev, r_prev) satisfying adjacent constraints:
                # L_i != L_prev and R_i != R_prev.
                for jdx, (l_prev, r_prev) in enumerate(valid_pairs):
                    if l_prev != l and r_prev != r:
                        if dp[i-1][jdx] < best_prev:
                            best_prev = dp[i-1][jdx]
                dp[i][idx] = cur_cost + best_prev
        
        # The answer is the minimum cost among all assignments for the last pair.
        result = min(dp[half-1])
        return result