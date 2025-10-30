from typing import List
import math

class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        m, n = len(coins), len(coins[0])
        # We allow at most 2 neutralizations, so we'll consider states 0,1,2.
        # Initialize dp with -infinity for all positions and states.
        dp = [[[-10**15] * 3 for _ in range(n)] for __ in range(m)]
        
        # Initialize start cell (0,0)
        # If the starting cell is non-negative, no need to neutralize:
        if coins[0][0] >= 0:
            dp[0][0][0] = coins[0][0]
        else:
            # Option without neutralization:
            dp[0][0][0] = coins[0][0]
            # Option neutralizing if available (>0 uses among 2 allowed):
            dp[0][0][1] = 0
        
        # Function to update dp for a given cell (i,j) from a predecessor state value and used neutralizations count.
        def update(i, j, prevVal, used):
            # At coins[i][j]
            v = coins[i][j]
            # Without neutralizing this cell:
            newVal = prevVal + v
            if newVal > dp[i][j][used]:
                dp[i][j][used] = newVal
            # If current cell is negative and we have a neutralization available (i.e., used < 2),
            # then we can choose to neutralize this negative cell.
            if v < 0 and used < 2:
                newUsed = used + 1
                newValNeutral = prevVal + 0  # because we avoid the loss
                if newValNeutral > dp[i][j][newUsed]:
                    dp[i][j][newUsed] = newValNeutral

        # Iterate through the grid and update dp values.
        for i in range(m):
            for j in range(n):
                # skip (0,0) since it is already processed.
                if i == 0 and j == 0:
                    continue
                
                # For each possible predecessor, and for each used neutralizations state.
                # Predecessor from top (i-1, j) if exists.
                if i > 0:
                    for used in range(3):
                        if dp[i-1][j][used] != -10**15:
                            update(i, j, dp[i-1][j][used], used)
                # Predecessor from left (i, j-1)
                if j > 0:
                    for used in range(3):
                        if dp[i][j-1][used] != -10**15:
                            update(i, j, dp[i][j-1][used], used)
        
        # The answer is the best we can have at (m-1, n-1) for any neutralization state.
        return max(dp[m-1][n-1])