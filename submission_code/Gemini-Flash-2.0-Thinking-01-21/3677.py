import math
from typing import List

class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        m = len(coins)
        n = len(coins[0])

        # dp[i][j][k] = maximum coins reaching cell (i, j) having used exactly k neutralizations.
        # k can be 0, 1, or 2.
        # Initialize DP table with negative infinity to represent unreachable states or
        # paths that haven't been calculated yet. Using -math.inf is appropriate as
        # coin amounts can be negative, and any valid path will have a value >= -500*500*1000,
        # which is much greater than -math.inf.
        dp = [[[ -math.inf for _ in range(3)] for _ in range(n)] for _ in range(m)]

        # Base Case: Starting cell (0, 0)
        # To reach (0, 0) using 0 neutralizations, the value is simply coins[0][0].
        dp[0][0][0] = coins[0][0]

        # If the starting cell contains a robber (negative value), we have the option
        # to use one neutralization on it right at the start.
        # If a negative cell is neutralized, its contribution to the total coins is 0.
        if coins[0][0] < 0:
            # Reaching (0, 0) using 1 neutralization (used on the starting cell itself).
            # The value contributed is 0.
            dp[0][0][1] = 0

        # Iterate through the grid, filling the DP table
        for i in range(m):
            for j in range(n):
                # The base case (0, 0) is handled separately above
                if i == 0 and j == 0:
                    continue

                val = coins[i][j]

                # Iterate through the possible number of neutralizations used to reach cell (i, j)
                for k in range(3): # k represents the count of neutralizations used *up to and including* cell (i, j)

                    # Option 1: Arrive at (i, j) having already used k neutralizations BEFORE,
                    # and DO NOT neutralize at cell (i, j).
                    # This implies that if coins[i][j] is negative, we take the loss.
                    # We could have arrived from the cell above (i-1, j) or the cell to the left (i, j-1),
                    # having accumulated maximum coins stored in dp[i-1][j][k] or dp[i][j-1][k].
                    # If the previous state was reachable (not -math.inf), we consider this path.
                    if i > 0 and dp[i-1][j][k] != -math.inf:
                        dp[i][j][k] = max(dp[i][j][k], dp[i-1][j][k] + val)
                    if j > 0 and dp[i][j-1][k] != -math.inf:
                         dp[i][j][k] = max(dp[i][j][k], dp[i][j-1][k] + val)

                    # Option 2: Arrive at (i, j) having already used k-1 neutralizations BEFORE,
                    # and DO neutralize at cell (i, j).
                    # This is only possible if the current cell's value is negative (a robber)
                    # and if we have neutralizations available (k > 0).
                    # We could have arrived from (i-1, j) or (i, j-1), having used k-1 neutralizations.
                    # If a negative cell is neutralized, its contribution to the total coins is 0.
                    # We need to check if k > 0 because we are transitioning from a state with k-1 neutralizations.
                    if val < 0 and k > 0:
                        if i > 0 and dp[i-1][j][k-1] != -math.inf:
                            dp[i][j][k] = max(dp[i][j][k], dp[i-1][j][k-1] + 0) # Add 0 as the value is neutralized
                        if j > 0 and dp[i][j-1][k-1] != -math.inf:
                            dp[i][j][k] = max(dp[i][j][k], dp[i][j-1][k-1] + 0) # Add 0 as the value is neutralized

        # The final answer is the maximum amount of coins upon reaching the bottom-right corner (m-1, n-1).
        # This maximum could be achieved using 0, 1, or 2 neutralizations in total.
        # We need to take the maximum value among all possible k states at the destination.
        return max(dp[m-1][n-1][0], dp[m-1][n-1][1], dp[m-1][n-1][2])