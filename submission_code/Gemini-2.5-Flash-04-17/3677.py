import math
from typing import List

class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        m = len(coins)
        n = len(coins[0])

        # dp[i][j][k] = max coins gained reaching cell (i, j) using exactly k neutralizations
        # k can be 0, 1, or 2, representing the total count of neutralizations used
        # on the path from (0, 0) to (i, j), inclusive.
        dp = [[[-math.inf] * 3 for _ in range(n)] for _ in range(m)]

        # Base case: starting cell (0, 0)
        current_value_00 = coins[0][0]

        # When at (0, 0), we can either not neutralize the robber or neutralize if it's a robber.
        if current_value_00 >= 0:
            # If coins[0][0] is non-negative, we gain the coins. No neutralization used yet.
            dp[0][0][0] = current_value_00
        else: # current_value_00 < 0
            # Option 1: Don't neutralize at (0,0). Uses 0 neutralizations in total up to (0,0).
            # We take the penalty (add the negative value).
            dp[0][0][0] = current_value_00
            # Option 2: Neutralize at (0,0). Uses 1 neutralization in total up to (0,0).
            # We avoid the penalty, effectively adding 0 instead of coins[0][0].
            # This is only possible for the state where k=1 neutralization is used.
            dp[0][0][1] = 0 # Cost 1 neut, gain 0

        # Fill the DP table for the rest of the grid cells
        for i in range(m):
            for j in range(n):
                # Skip the base case cell (0, 0) as it's already initialized
                if i == 0 and j == 0:
                    continue

                current_value = coins[i][j]

                # k represents the *total* number of neutralizations used *up to and including* cell (i, j).
                # Iterate through the possible number of neutralizations used to reach (i, j)
                for k in range(3):

                    # Consider paths arriving from the cell directly above (i-1, j)
                    if i > 0:
                        # Sub-option 1.1: Robot arrived at (i-1, j) having used k neutralizations,
                        # and *does not* neutralize at the current cell (i, j).
                        # The total number of neutralizations used up to (i, j) remains k.
                        # This transition is valid if dp[i-1][j][k] was a reachable state.
                        if dp[i-1][j][k] != -math.inf:
                            dp[i][j][k] = max(dp[i][j][k], dp[i-1][j][k] + current_value)

                        # Sub-option 1.2: Robot arrived at (i-1, j) having used k-1 neutralizations,
                        # and *does* neutralize at the current cell (i, j).
                        # The total number of neutralizations used up to (i, j) becomes k.
                        # This is only possible if the current cell contains a robber (current_value < 0),
                        # and we have used exactly k-1 neuts on the path before (k > 0),
                        # and dp[i-1][j][k-1] was a reachable state.
                        if current_value < 0 and k > 0 and dp[i-1][j][k-1] != -math.inf:
                            # Neutralizing the robber means adding 0 instead of the negative value.
                            dp[i][j][k] = max(dp[i][j][k], dp[i-1][j][k-1] + 0)

                    # Consider paths arriving from the cell directly left (i, j-1)
                    if j > 0:
                        # Sub-option 2.1: Robot arrived at (i, j-1) having used k neutralizations,
                        # and *does not* neutralize at the current cell (i, j).
                        # The total number of neutralizations used up to (i, j) remains k.
                        # This transition is valid if dp[i][j-1][k] was a reachable state.
                        if dp[i][j-1][k] != -math.inf:
                            dp[i][j][k] = max(dp[i][j][k], dp[i][j-1][k] + current_value)

                        # Sub-option 2.2: Robot arrived at (i, j-1) having used k-1 neutralizations,
                        # and *does* neutralize at the current cell (i, j).
                        # The total number of neutralizations used up to (i, j) becomes k.
                        # This is only possible if the current cell contains a robber (current_value < 0),
                        # and we have used exactly k-1 neuts on the path before (k > 0),
                        # and dp[i][j-1][k-1] was a reachable state.
                        if current_value < 0 and k > 0 and dp[i][j-1][k-1] != -math.inf:
                            # Neutralizing the robber means adding 0 instead of the negative value.
                            dp[i][j][k] = max(dp[i][j][k], dp[i][j-1][k-1] + 0)


        # The final answer is the maximum coins achieved upon reaching the bottom-right corner (m-1, n-1),
        # considering any possible number of neutralizations used on the path (0, 1, or at most 2).
        result = -math.inf
        # We check the maximum value among all possible neutralization counts (0, 1, or 2) at the destination cell.
        for k in range(3):
             result = max(result, dp[m-1][n-1][k])

        return result