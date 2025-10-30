class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        dp = [[0]*(n+1) for _ in range(4)]
        dp[0][0] = 1
        for i in range(1, 4):
            for j in range(n+1):
                dp[i][j] = dp[i-1][j]
                if j >= 1:
                    dp[i][j] += dp[i][j-1]
                if j > limit:
                    dp[i][j] -= dp[i-1][j-limit-1]
        return dp[3][n]