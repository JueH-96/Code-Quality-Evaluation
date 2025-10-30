class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        mod = 10**9 + 7
        n = len(s)
        s = s + s
        t = t[::-1]
        dp = [[0]*(k+1) for _ in range(2*n+1)]
        dp[0][0] = 1
        for i in range(1, 2*n+1):
            dp[i][0] = 1
            for j in range(1, min(i, k)+1):
                dp[i][j] = dp[i-1][j]
                if s[i-1] == t[j-1]:
                    dp[i][j] += dp[i-1][j-1]
                dp[i][j] %= mod
        ans = 0
        for i in range(n, 2*n):
            ans += dp[i][k]
            ans %= mod
        return ans