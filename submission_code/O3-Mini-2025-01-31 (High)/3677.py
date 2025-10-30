from typing import List
import math

class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        m, n = len(coins), len(coins[0])
        # We use 3 states: 0, 1, or 2 neutralizations used.
        # Initialize dp with -infinity for unreachable states.
        dp = [[[-math.inf] * 3 for _ in range(n)] for _ in range(m)]
        
        # Initialize the starting cell (0,0)
        start_val = coins[0][0]
        if start_val >= 0:
            dp[0][0][0] = start_val
        else:
            # Option 1: Do not use the ability
            dp[0][0][0] = start_val
            # Option 2: Use ability to neutralize the robbery
            dp[0][0][1] = 0
        
        # Directions: right, down
        for i in range(m):
            for j in range(n):
                for used in range(3):
                    if dp[i][j][used] == -math.inf:
                        continue
                    # Try moving right
                    if j + 1 < n:
                        nxt = coins[i][j+1]
                        # If the cell value is non-negative, just add it.
                        if nxt >= 0:
                            dp[i][j+1][used] = max(dp[i][j+1][used], dp[i][j][used] + nxt)
                        else:
                            # Option without neutralizing: add the negative value.
                            dp[i][j+1][used] = max(dp[i][j+1][used], dp[i][j][used] + nxt)
                            # Option with neutralizing if available.
                            if used < 2:
                                dp[i][j+1][used+1] = max(dp[i][j+1][used+1], dp[i][j][used] + 0)
                    
                    # Try moving down
                    if i + 1 < m:
                        nxt = coins[i+1][j]
                        if nxt >= 0:
                            dp[i+1][j][used] = max(dp[i+1][j][used], dp[i][j][used] + nxt)
                        else:
                            dp[i+1][j][used] = max(dp[i+1][j][used], dp[i][j][used] + nxt)
                            if used < 2:
                                dp[i+1][j][used+1] = max(dp[i+1][j][used+1], dp[i][j][used] + 0)
        
        # The answer is the maximum coins among all states at the target cell.
        return max(dp[m-1][n-1])