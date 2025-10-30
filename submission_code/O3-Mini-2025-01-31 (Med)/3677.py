from typing import List

class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        m, n = len(coins), len(coins[0])
        NEG_INF = float("-inf")
        
        # dp[i][j][k] : maximum coins reached at cell (i,j) using exactly k neutralizations.
        # k can be 0, 1, 2.
        dp = [[[NEG_INF] * 3 for _ in range(n)] for _ in range(m)]
        
        # Initialization for cell (0,0)
        if coins[0][0] >= 0:
            dp[0][0][0] = coins[0][0]
        else:
            # Option 1: do not neutralize the negative coin.
            dp[0][0][0] = coins[0][0]
            # Option 2: neutralize it if possible (using one neutralization).
            dp[0][0][1] = 0

        # Function to update dp for cell (i, j) with coin value cell_val coming from previous best value prev_val,
        # and current neutralizations used so far = k_used.
        # We update the dp[i][j] for new state considering the current cell.
        for i in range(m):
            for j in range(n):
                # Skip the starting cell since it is already processed.
                if i == 0 and j == 0:
                    continue

                # For each cell, consider the best from top or left.
                # We'll use a temporary array for the dp state for current cell.
                cur_dp = [NEG_INF, NEG_INF, NEG_INF]
                # Check from top.
                if i > 0:
                    for k in range(3):
                        if dp[i-1][j][k] != NEG_INF:
                            # Try to use transition from top.
                            prev = dp[i-1][j][k]
                            if coins[i][j] >= 0:
                                # Just add coin gain.
                                cur_dp[k] = max(cur_dp[k], prev + coins[i][j])
                            else:
                                # Option 1: Do not neutralize this negative cell.
                                cur_dp[k] = max(cur_dp[k], prev + coins[i][j])
                                # Option 2: neutralize this cell if possible.
                                if k < 2:
                                    cur_dp[k+1] = max(cur_dp[k+1], prev)
                # Check from left.
                if j > 0:
                    for k in range(3):
                        if dp[i][j-1][k] != NEG_INF:
                            prev = dp[i][j-1][k]
                            if coins[i][j] >= 0:
                                cur_dp[k] = max(cur_dp[k], prev + coins[i][j])
                            else:
                                cur_dp[k] = max(cur_dp[k], prev + coins[i][j])
                                if k < 2:
                                    cur_dp[k+1] = max(cur_dp[k+1], prev)
                
                # Update dp[i][j] with the computed values.
                dp[i][j] = cur_dp
        
        # The answer is the maximum profit at the bottom-right cell using any number of neutralizations (0,1,2)
        result = max(dp[m-1][n-1])
        return result