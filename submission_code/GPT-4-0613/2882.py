class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        dp = [0] * (n + 1)
        dp[0] = 1
        for i in range(1, int(n**(1/x)) + 1):
            for j in range(n, i**x - 1, -1):
                dp[j] = (dp[j] + dp[j - i**x]) % MOD
        return dp[n]