from typing import List

class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        """
        Calculates the maximum profit a robot can gain moving from (0,0) to (m-1,n-1)
        in a grid, with the ability to neutralize up to 2 robbers.
        """
        m = len(coins)
        n = len(coins[0])
        
        # dp[i][j][k]: max profit to reach (i, j) using k neutralizations (k=0, 1, 2)
        dp = [[[-float('inf')] * 3 for _ in range(n)] for _ in range(m)]
        
        # Base case at the starting cell (0, 0)
        c00 = coins[0][0]
        if c00 >= 0:
            dp[0][0][0] = c00
        else: # c00 < 0 (robber)
            # Option 1: Don't neutralize (k=0)
            dp[0][0][0] = c00
            # Option 2: Neutralize (k=1)
            dp[0][0][1] = 0
        
        # Fill the DP table by iterating through the grid
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                
                # Max profit from previous cells (up or left) for each k
                prev_profits = [-float('inf')] * 3
                if i > 0:
                    for k in range(3):
                        prev_profits[k] = max(prev_profits[k], dp[i-1][j][k])
                if j > 0:
                    for k in range(3):
                        prev_profits[k] = max(prev_profits[k], dp[i][j-1][k])
                
                current_coin = coins[i][j]
                
                for k in range(3):
                    if current_coin >= 0:
                        # Cannot neutralize a positive coin cell.
                        # Must come from a path with k neutralizations.
                        if prev_profits[k] != -float('inf'):
                            dp[i][j][k] = prev_profits[k] + current_coin
                    else: # current_coin < 0 (robber)
                        # Option 1: Don't neutralize. Path came with k neutralizations.
                        profit_no_neutralize = -float('inf')
                        if prev_profits[k] != -float('inf'):
                            profit_no_neutralize = prev_profits[k] + current_coin
                        
                        # Option 2: Neutralize. Path came with k-1 neutralizations.
                        profit_neutralize = -float('inf')
                        if k > 0 and prev_profits[k-1] != -float('inf'):
                            profit_neutralize = prev_profits[k-1]
                            
                        dp[i][j][k] = max(profit_no_neutralize, profit_neutralize)

        # The final answer is the max profit at the destination over all neutralization counts.
        result = max(dp[m-1][n-1])
        
        return result