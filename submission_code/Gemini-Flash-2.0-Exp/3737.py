class Solution:
    def minCost(self, n: int, cost: List[List[int]]) -> int:
        dp = [[float('inf')] * 3 for _ in range(n)]

        for i in range(3):
            dp[0][i] = cost[0][i]

        for i in range(1, n):
            for j in range(3):
                for k in range(3):
                    if j != k:
                        if i != n // 2 and (i < n // 2 and j == cost[n - 1 - i].index(min(cost[n - 1 - i])) or i > n // 2 -1 and j == cost[n - 1 - i].index(min(cost[n - 1 - i]))):
                            dp[i][j] = min(dp[i][j], dp[i - 1][k] + cost[i][j])
                        elif i == n // 2 and j != cost[n - 1 - i].index(min(cost[n - 1 - i])):
                            dp[i][j] = min(dp[i][j], dp[i - 1][k] + cost[i][j])
                        elif i < n // 2 and j != cost[n - 1 - i].index(min(cost[n - 1 - i])):
                            dp[i][j] = min(dp[i][j], dp[i - 1][k] + cost[i][j])
                        elif i > n // 2 -1 and j != cost[n - 1 - i].index(min(cost[n - 1 - i])):
                            dp[i][j] = min(dp[i][j], dp[i - 1][k] + cost[i][j])
                        else:
                            dp[i][j] = min(dp[i][j], dp[i - 1][k] + cost[i][j])
        
        ans = float('inf')
        for i in range(3):
            ans = min(ans, dp[n - 1][i])
        return ans