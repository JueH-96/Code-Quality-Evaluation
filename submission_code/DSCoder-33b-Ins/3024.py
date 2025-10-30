class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        m = len(t)
        dp = [[0]*(k+1) for _ in range(m+1)]
        dp[0] = [1]*(k+1)
        for i in range(1, m+1):
            for j in range(i, n-(m-i)+1):
                dp[i][j] = (dp[i][j-1] + dp[i-1][j-1]*(s[j-1] == t[i-1])) % MOD
        return dp[m][k]