class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        dp = [0] * (n + 1)
        dp[0] = 1
        for i in range(1, n + 1):
            dp[i] = dp[i - 1] * 2 % MOD
            if i - 3 >= 0:
                dp[i] = (dp[i] - dp[i - 3] + MOD) % MOD
        return dp[n]