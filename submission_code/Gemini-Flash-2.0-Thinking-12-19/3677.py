from typing import List
import math # Needed for -math.inf

class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        m = len(coins)
        n = len(coins[0])

        # dp[i][j][k] stores the maximum coins to reach cell (i, j) using exactly k neutralizations
        # k can be 0, 1, or 2.
        # Initialize with a value representing unreachable state.
        dp = [[[-math.inf] * 3 for _ in range(n)] for _ in range(m)]

        # Base case: (0, 0)
        start_val = coins[0][0]
        if start_val >= 0:
            # If starting cell has non-negative coins, we cannot "neutralize" it.
            # We start with start_val coins and 0 neutralizations used at (0,0).
            dp[0][0][0] = start_val
        else: # start_val < 0 (is a robber)
            # Option 1: Don't neutralize the robber at (0,0).
            # We start with start_val coins and 0 neutralizations used at (0,0).
            dp[0][0][0] = start_val
            # Option 2: Neutralize the robber at (0,0).
            # We start with 0 coins from this cell (cost avoided) and use 1 neutralization.
            dp[0][0][1] = 0

        # Fill the DP table
        # Iterate through cells row by row, column by column, starting from (0,1) or (1,0)
        for i in range(m):
            for j in range(n):
                # Skip the base case which is already handled
                if i == 0 and j == 0:
                    continue

                current_val = coins[i][j]

                # Iterate through the number of neutralizations used to reach (i, j).
                # k represents the *total* number of neutralizations used *up to and including* cell (i, j).
                for k in range(3):
                    
                    # Option 1: Arrive at (i, j) having NOT neutralized current_val
                    # If we don't neutralize coins[i][j], the number of neutralizations used *before* reaching (i,j)
                    # must be the same as the number used *at* (i,j), which is k.
                    # This path must come from (i-1, j) or (i, j-1) having used 'k' neutralizations already.
                    val_no_neutralize = -math.inf
                    
                    # Check path from above (i-1, j) with k neutralizations
                    if i > 0 and dp[i-1][j][k] != -math.inf:
                        val_no_neutralize = max(val_no_neutralize, dp[i-1][j][k] + current_val)
                    
                    # Check path from left (i, j-1) with k neutralizations
                    if j > 0 and dp[i][j-1][k] != -math.inf:
                        val_no_neutralize = max(val_no_neutralize, dp[i][j-1][k] + current_val)

                    # Option 2: Arrive at (i, j) having neutralized current_val
                    # This is only possible if current_val < 0 (it's a robber) AND we have neutralizations available
                    # to use one at this cell. If we use the k-th neutralization at this cell,
                    # the path must have come from (i-1, j) or (i, j-1) having used 'k-1' neutralizations already.
                    # This implies k must be > 0.
                    val_neutralize = -math.inf
                    if current_val < 0 and k > 0: # Can only neutralize if it's a robber and we have neutralizations left (k > 0)
                        # Check path from above (i-1, j) with k-1 neutralizations
                        if i > 0 and dp[i-1][j][k-1] != -math.inf:
                             # If neutralized, the value added from coins[i][j] is effectively 0.
                             # We just take the maximum coins accumulated before reaching (i,j) using k-1 neutralizations.
                             val_neutralize = max(val_neutralize, dp[i-1][j][k-1])

                        # Check path from left (i, j-1) with k-1 neutralizations
                        if j > 0 and dp[i][j-1][k-1] != -math.inf:
                            # If neutralized, the value added from coins[i][j] is effectively 0.
                            val_neutralize = max(val_neutralize, dp[i][j-1][k-1])

                    # Update dp[i][j][k] with the maximum of the two valid options.
                    # If both options resulted in -math.inf, dp[i][j][k] remains -math.inf, indicating this state is unreachable.
                    dp[i][j][k] = max(val_no_neutralize, val_neutralize)

        # The final answer is the maximum coins accumulated at the bottom-right corner (m-1, n-1)
        # considering any number of neutralizations used (0, 1, or 2).
        result = -math.inf
        for k in range(3):
            result = max(result, dp[m-1][n-1][k])

        # The problem guarantees a path exists (m, n >= 1 and only right/down moves).
        # At least the state dp[0][0][0] or dp[0][0][1] is initialized.
        # A path exists from (0,0) to (m-1, n-1).
        # The DP structure should explore all such paths considering neutralizations.
        # Thus, result should not remain -math.inf unless there's an issue, which is unlikely given the problem constraints.
        # We can potentially add a check or return a default value if result is still -inf, but it's likely unnecessary.

        return result