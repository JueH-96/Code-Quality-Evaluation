class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        dp = [[float('inf')]*(n+1) for _ in range(n+1)]
        for i in range(n+1):
            dp[i][0] = 0
        for i in range(1, n+1):
            for j in range(1, n+1):
                dp[i][j] = min(dp[i-1][j-1]+cost[i-1], dp[max(0, i-1-time[i-1])][j-1])
        return dp[n][n]