from typing import List

class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        m = len(coins)
        n = len(coins[0])
        INF = float('-inf')
        dp = [[ [INF for _ in range(3)] for __ in range(n)] for ___ in range(m)]
        
        # Initialize the starting cell
        if coins[0][0] >= 0:
            dp[0][0][2] = coins[0][0]
        else:
            dp[0][0][2] = coins[0][0]
            dp[0][0][1] = 0  # Using one neutralizer here
        
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                current = coins[i][j]
                for k in range(3):
                    if current >= 0:
                        max_val = INF
                        if i > 0 and dp[i-1][j][k] > max_val:
                            max_val = dp[i-1][j][k]
                        if j > 0 and dp[i][j-1][k] > max_val:
                            max_val = dp[i][j-1][k]
                        if max_val != INF:
                            dp[i][j][k] = max_val + current
                        else:
                            dp[i][j][k] = INF
                    else:
                        # Option 1: Not using neutralizer
                        max_not_use = INF
                        if i > 0 and dp[i-1][j][k] > max_not_use:
                            max_not_use = dp[i-1][j][k]
                        if j > 0 and dp[i][j-1][k] > max_not_use:
                            max_not_use = dp[i][j-1][k]
                        option1 = max_not_use + current if max_not_use != INF else INF
                        
                        # Option 2: Using neutralizer
                        option2 = INF
                        if k < 2:
                            prev_k = k + 1
                            max_use = INF
                            if i > 0 and dp[i-1][j][prev_k] > max_use:
                                max_use = dp[i-1][j][prev_k]
                            if j > 0 and dp[i][j-1][prev_k] > max_use:
                                max_use = dp[i][j-1][prev_k]
                            if max_use != INF:
                                option2 = max_use + 0
                        
                        candidates = []
                        if option1 != INF:
                            candidates.append(option1)
                        if option2 != INF:
                            candidates.append(option2)
                        if candidates:
                            dp[i][j][k] = max(candidates)
                        else:
                            dp[i][j][k] = INF
        
        final_val = max(dp[m-1][n-1][0], dp[m-1][n-1][1], dp[m-1][n-1][2])
        return int(final_val) if final_val != INF else 0