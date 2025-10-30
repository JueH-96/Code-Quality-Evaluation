class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        dp = [[0] * (n + 1) for _ in range(limit + 1)]
        dp[0][0] = 1
        for i in range(1, limit + 1):
            for j in range(n + 1):
                dp[i][j] = dp[i - 1][j]
                if j >= i:
                    dp[i][j] += dp[i][j - i]
        return dp[limit][n]