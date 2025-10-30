import collections
from typing import List

class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        m, n = len(coins), len(coins[0])
        
        # dp[i][j][k] represents the maximum profit to reach cell (i, j)
        # having used 'k' neutralizations on the path so far.
        # 'k' can be 0, 1, or 2.
        # Initialize with float('-inf') to correctly handle unreachable states and find maximums.
        dp = [[[float('-inf')] * 3 for _ in range(n)] for _ in range(m)]

        # Base case for the starting cell (0, 0)
        val_00 = coins[0][0]
        
        # dp[0][0][0]: Profit if no neutralization is used at (0,0).
        dp[0][0][0] = val_00
        
        # dp[0][0][1]: Profit if one neutralization is used on the path.
        # If val_00 is negative, we can neutralize it for 0 profit from this cell,
        # consuming one neutralization.
        # If val_00 is positive/zero, we don't *need* to neutralize it.
        # In this context, reaching dp[0][0][1] (meaning k=1) means that if we had to use a neutralization
        # at this cell (it being the first one on the path), we would. Since we didn't,
        # the profit is just val_00, and the k=1 state is simply reachable, carrying the value.
        dp[0][0][1] = 0 if val_00 < 0 else val_00
        
        # dp[0][0][2]: Similar logic for two neutralizations used.
        # For a single cell, we can only use one neutralization. The profit from this cell is 0 if negative,
        # or val_00 if positive/zero. This state implies that we are on a path where 2 neutralizations
        # are ultimately consumed, and this cell is one part of it (potentially not consuming a neut here).
        dp[0][0][2] = 0 if val_00 < 0 else val_00

        # Iterate through the grid to fill the DP table
        for i in range(m):
            for j in range(n):
                # Skip the starting cell (0, 0) as it's already initialized
                if i == 0 and j == 0:
                    continue

                current_coin = coins[i][j]

                # Calculate dp[i][j][k] for each possible number of neutralizations used ('k')
                for k in range(3): # 'k' can be 0, 1, or 2
                    # Option A: Arrive at (i,j) and DO NOT neutralize current_coin.
                    # The number of neutralizations 'k' remains the same as in the previous cell.
                    
                    # Max profit from paths ending at (i,j) without neutralizing current_coin
                    max_profit_no_neut_option = float('-inf')

                    # Consider path from the cell directly above (i-1, j)
                    if i > 0 and dp[i-1][j][k] != float('-inf'):
                        max_profit_no_neut_option = max(max_profit_no_neut_option, dp[i-1][j][k] + current_coin)
                    
                    # Consider path from the cell directly to the left (i, j-1)
                    if j > 0 and dp[i][j-1][k] != float('-inf'):
                        max_profit_no_neut_option = max(max_profit_no_neut_option, dp[i][j-1][k] + current_coin)
                    
                    # Initialize dp[i][j][k] with the best profit from this option
                    dp[i][j][k] = max_profit_no_neut_option

                    # Option B: Arrive at (i,j) and DO neutralize current_coin.
                    # This is only possible if current_coin is negative (a robber) AND
                    # we have a neutralization budget available (k > 0, meaning we increment from k-1 to k).
                    # If we neutralize, the value gained from current_coin is 0.
                    # The state we come from would have used (k-1) neutralizations.
                    if current_coin < 0 and k > 0:
                        # Max profit from paths ending at (i,j) by neutralizing current_coin
                        max_profit_with_neut_option = float('-inf')

                        # Consider path from the cell directly above (i-1, j) using (k-1) neutralizations
                        if i > 0 and dp[i-1][j][k-1] != float('-inf'):
                            max_profit_with_neut_option = max(max_profit_with_neut_option, dp[i-1][j][k-1] + 0)
                        
                        # Consider path from the cell directly to the left (i, j-1) using (k-1) neutralizations
                        if j > 0 and dp[i][j-1][k-1] != float('-inf'):
                            max_profit_with_neut_option = max(max_profit_with_neut_option, dp[i][j-1][k-1] + 0)
                        
                        # Update dp[i][j][k] with the maximum profit considering both options (neutralize or not)
                        dp[i][j][k] = max(dp[i][j][k], max_profit_with_neut_option)

        # The final answer is the maximum profit at the bottom-right corner (m-1, n-1)
        # among all possible neutralization counts (0, 1, or 2).
        return max(dp[m-1][n-1][0], dp[m-1][n-1][1], dp[m-1][n-1][2])