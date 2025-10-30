from typing import List
import sys

class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        m = len(coins)
        n = len(coins[0])
        INF = float('-inf')
        dp = [[[INF for _ in range(3)] for __ in range(n)] for ___ in range(m)]
        
        # Initialize the starting cell (0,0)
        dp[0][0][2] = coins[0][0]
        dp[0][0][1] = abs(coins[0][0])
        # dp[0][0][0] remains -inf as it's impossible to have 0 neutralizations left here
        
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                # For each possible k (0, 1, 2)
                for k in range(3):
                    max_val = INF
                    # Check top direction (i-1, j)
                    if i > 0:
                        # Option 1: not use neutralization here
                        option1 = dp[i-1][j][k] + coins[i][j]
                        # Option 2: use neutralization here if possible
                        option2 = INF
                        if k < 2:
                            option2 = dp[i-1][j][k+1] + abs(coins[i][j])
                        current_max = max(option1, option2)
                        if current_max > max_val:
                            max_val = current_max
                    # Check left direction (i, j-1)
                    if j > 0:
                        # Option 1: not use neutralization here
                        option1_left = dp[i][j-1][k] + coins[i][j]
                        # Option 2: use neutralization here if possible
                        option2_left = INF
                        if k < 2:
                            option2_left = dp[i][j-1][k+1] + abs(coins[i][j])
                        current_max_left = max(option1_left, option2_left)
                        if current_max_left > max_val:
                            max_val = current_max_left
                    dp[i][j][k] = max_val
        
        # The answer is the maximum of the three possibilities at the bottom-right cell
        return max(dp[m-1][n-1][0], dp[m-1][n-1][1], dp[m-1][n-1][2])