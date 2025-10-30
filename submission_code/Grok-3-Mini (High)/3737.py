from typing import List
import math

class Solution:
    def minCost(self, n: int, cost: List[List[int]]) -> int:
        k = n // 2  # Number of pairs
        
        # Initialize dp for the first pair (pair 0)
        dp_prev = [[float('inf') for _ in range(3)] for _ in range(3)]
        for cL in range(3):
            for cR in range(3):
                if cL != cR:
                    dp_prev[cL][cR] = cost[0][cL] + cost[n-1][cR]
        
        # For each additional pair from 1 to k-1
        if k > 1:
            for pair_idx in range(1, k):
                curr_dp = [[float('inf') for _ in range(3)] for _ in range(3)]
                for cL in range(3):  # Current left color
                    for cR in range(3):  # Current right color
                        if cL != cR:  # Ensure colors within pair are different
                            for prev_L in range(3):  # Previous left color
                                if cL != prev_L:  # Constraint: current left != previous left
                                    for prev_R in range(3):  # Previous right color
                                        if cR != prev_R:  # Constraint: current right != previous right
                                            curr_dp[cL][cR] = min(curr_dp[cL][cR], dp_prev[prev_L][prev_R] + cost[pair_idx][cL] + cost[n-1 - pair_idx][cR])
                # Update dp_prev to curr_dp for the next iteration
                dp_prev = curr_dp
        
        # The minimum cost is the minimum value in dp_prev
        ans = min(dp_prev[i][j] for i in range(3) for j in range(3))
        return int(ans)