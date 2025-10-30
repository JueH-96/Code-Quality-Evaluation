class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        dp = [0]*(n+1)
        dp[0] = 1
        for i in range(1, n+1):
            dp[i] = dp[i-1]*2 % MOD if i < 2**x else (dp[i-1] + dp[i-2**x]) % MOD
        return dp[n]