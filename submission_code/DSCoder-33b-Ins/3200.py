class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        dp = [[0]*8 for _ in range(n+1)]
        dp[0][0] = 1
        for i in range(1, n+1):
            for j in range(8):
                dp[i][j] = dp[i-1][j]
                if j > 0:
                    dp[i][j] += dp[i-1][j-1] * (26-(j-1))
                if i > 1:
                    dp[i][j] += dp[i-2][j] * max(0, j-2)
                dp[i][j] %= MOD
        return dp[n][7]