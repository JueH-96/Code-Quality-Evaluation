class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        mod = 10**9 + 7
        n = len(s)
        dp = [[0]*(k+1) for _ in range(n+1)]
        dp[0][0] = 1
        for i in range(1, n+1):
            dp[i][0] = 1
            for j in range(1, k+1):
                dp[i][j] = (dp[i-1][j] + dp[i-1][j-1]*(s[i-1]==t[i-1])) % mod
        return dp[n][k]