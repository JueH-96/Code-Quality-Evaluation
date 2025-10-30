class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        dp = [0] * (n + 1)
        dp[0] = 1

        i = 1
        while i**x <= n:
            for j in range(n, i**x - 1, -1):
                dp[j] = (dp[j] + dp[j - i**x]) % MOD
            i += 1

        return dp[n]