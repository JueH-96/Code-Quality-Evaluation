class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        m = len(coins)
        n = len(coins[0])
        INF = -10**18
        dp = [[[INF] * 3 for _ in range(n)] for __ in range(m)]
        
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
                    candidates_no_neutralize = []
                    candidates_neutralize = []
                    
                    if i > 0:
                        candidates_no_neutralize.append(dp[i-1][j][k])
                        if k >= 1 and coins[i][j] < 0:
                            candidates_neutralize.append(dp[i-1][j][k-1])
                    if j > 0:
                        candidates_no_neutralize.append(dp[i][j-1][k])
                        if k >= 1 and coins[i][j] < 0:
                            candidates_neutralize.append(dp[i][j-1][k-1])
                    
                    if coins[i][j] >= 0:
                        if candidates_no_neutralize:
                            best_prev = max(candidates_no_neutralize)
                            dp[i][j][k] = best_prev + coins[i][j]
                    else:
                        option1 = INF
                        if candidates_no_neutralize:
                            option1 = max(candidates_no_neutralize) + coins[i][j]
                        option2 = INF
                        if candidates_neutralize:
                            option2 = max(candidates_neutralize)
                        dp[i][j][k] = max(option1, option2)
        
        return max(dp[m-1][n-1])