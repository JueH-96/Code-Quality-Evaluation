from typing import List

class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        m, n = len(coins), len(coins[0])
        # dp[i][j][k]: max coins at (i,j) having used k neutralizations (k=0,1,2)
        # Initialize to very small
        NEG_INF = -10**18
        dp = [[[NEG_INF]*3 for _ in range(n)] for __ in range(m)]
        
        # Initialize start cell (0,0)
        v0 = coins[0][0]
        if v0 >= 0:
            dp[0][0][0] = v0
        else:
            # take loss without neutralizing
            dp[0][0][0] = v0
            # or neutralize here (use 1 neutralization, gain 0)
            dp[0][0][1] = 0
        
        # Fill first row
        for j in range(1, n):
            v = coins[0][j]
            for k in range(3):
                prev = dp[0][j-1][k]
                if prev == NEG_INF:
                    continue
                # not neutralizing at (0,j)
                dp[0][j][k] = max(dp[0][j][k], prev + v)
                # if robber and have spare neutralization
                if v < 0 and k+1 < 3:
                    dp[0][j][k+1] = max(dp[0][j][k+1], prev)
        
        # Fill first column
        for i in range(1, m):
            v = coins[i][0]
            for k in range(3):
                prev = dp[i-1][0][k]
                if prev == NEG_INF:
                    continue
                dp[i][0][k] = max(dp[i][0][k], prev + v)
                if v < 0 and k+1 < 3:
                    dp[i][0][k+1] = max(dp[i][0][k+1], prev)
        
        # Fill the rest
        for i in range(1, m):
            for j in range(1, n):
                v = coins[i][j]
                for k in range(3):
                    # transition without neutralizing here
                    prev_best = max(dp[i-1][j][k], dp[i][j-1][k])
                    if prev_best != NEG_INF:
                        dp[i][j][k] = max(dp[i][j][k], prev_best + v)
                    # transition by using a neutralization here if v<0
                    if v < 0 and k > 0:
                        prev_best2 = max(dp[i-1][j][k-1], dp[i][j-1][k-1])
                        if prev_best2 != NEG_INF:
                            dp[i][j][k] = max(dp[i][j][k], prev_best2)
        
        # Answer is best at bottom-right with up to 2 neutralizations
        return max(dp[m-1][n-1])