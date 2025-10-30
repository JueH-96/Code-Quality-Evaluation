class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        MOD = 10**9 + 7
        dp = [0] * (n + 1)
        dp[0] = 1
        for _ in range(k):
            new_dp = dp[:]
            for i in range(1, n + 1):
                new_dp[i] = (new_dp[i - 1] + dp[i]) % MOD
            dp = new_dp
        return dp[-1]