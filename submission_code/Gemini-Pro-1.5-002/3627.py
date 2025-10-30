class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n = len(moveTime)
        m = len(moveTime[0])
        dp = [[float('inf')] * m for _ in range(n)]
        dp[0][0] = 0
        for i in range(n):
            for j in range(m):
                if i > 0:
                    dp[i][j] = min(dp[i][j], max(dp[i - 1][j] + 1, moveTime[i][j]))
                if j > 0:
                    dp[i][j] = min(dp[i][j], max(dp[i][j - 1] + 1, moveTime[i][j]))
        return dp[n - 1][m - 1]