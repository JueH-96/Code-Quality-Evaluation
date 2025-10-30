class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        if n < 4:
            return 0
        dp = [0] * (n + 1)
        dp[4] = 24
        for i in range(5, n + 1):
            dp[i] = (26 * dp[i - 1] - 24 * dp[i - 4]) % MOD
        return dp[n]