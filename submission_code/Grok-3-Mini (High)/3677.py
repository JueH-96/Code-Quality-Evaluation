from typing import List

class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        m = len(coins)
        n = len(coins[0])
        INF = -1000000000
        dp = [[[INF for _ in range(3)] for _ in range(n)] for _ in range(m)]
        
        # Set the starting point
        dp[0][0][0] = coins[0][0]
        if coins[0][0] < 0:
            dp[0][0][1] = 0  # Neutralized
        
        # Iterate through all cells in row-major order
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue  # Already set
                for k in range(3):  # k = 0, 1, 2
                    # Not neutralize option
                    vals_pred_same_k = []
                    if i > 0:
                        vals_pred_same_k.append(dp[i-1][j][k])
                    if j > 0:
                        vals_pred_same_k.append(dp[i][j-1][k])
                    max_pred_same_k = max(vals_pred_same_k) if vals_pred_same_k else INF
                    val_not_neutralize = coins[i][j] + max_pred_same_k
                    
                    # Neutralize option
                    if coins[i][j] < 0 and k >= 1:
                        vals_pred_k_minus_1 = []
                        if i > 0:
                            vals_pred_k_minus_1.append(dp[i-1][j][k-1])
                        if j > 0:
                            vals_pred_k_minus_1.append(dp[i][j-1][k-1])
                        max_pred_k_minus_1 = max(vals_pred_k_minus_1) if vals_pred_k_minus_1 else INF
                        val_neutralize = 0 + max_pred_k_minus_1
                    else:
                        val_neutralize = INF
                    
                    # Set dp[i][j][k] to the maximum of the two options
                    dp[i][j][k] = max(val_not_neutralize, val_neutralize)
        
        # The answer is the maximum over k at the bottom-right corner
        return max(dp[m-1][n-1][0], dp[m-1][n-1][1], dp[m-1][n-1][2])