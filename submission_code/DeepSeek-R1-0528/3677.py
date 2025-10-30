class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        m = len(coins)
        n = len(coins[0])
        dp = [[[-10**18] * 3 for _ in range(n)] for __ in range(m)]
        
        if coins[0][0] >= 0:
            dp[0][0][0] = coins[0][0]
        else:
            dp[0][0][0] = coins[0][0]
            dp[0][0][1] = 0
        
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                for k in range(3):
                    best_prev = -10**18
                    if i > 0:
                        best_prev = max(best_prev, dp[i-1][j][k])
                    if j > 0:
                        best_prev = max(best_prev, dp[i][j-1][k])
                    
                    if coins[i][j] >= 0:
                        if best_prev != -10**18:
                            dp[i][j][k] = best_prev + coins[i][j]
                    else:
                        option1 = best_prev + coins[i][j] if best_prev != -10**18 else -10**18
                        option2 = -10**18
                        if k >= 1:
                            best_prev_k1 = -10**18
                            if i > 0:
                                best_prev_k1 = max(best_prev_k1, dp[i-1][j][k-1])
                            if j > 0:
                                best_prev_k1 = max(best_prev_k1, dp[i][j-1][k-1])
                            if best_prev_k1 != -10**18:
                                option2 = best_prev_k1
                        dp[i][j][k] = max(option1, option2)
        
        return max(dp[m-1][n-1])