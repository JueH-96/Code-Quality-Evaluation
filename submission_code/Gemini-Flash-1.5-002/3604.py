class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = 10**9 + 7

        # dp[i][j] represents the number of ways to assign i performers to j stages
        dp = [[0] * (x + 1) for _ in range(n + 1)]
        dp[0][0] = 1

        for i in range(1, n + 1):
            for j in range(1, min(i, x) + 1):
                dp[i][j] = (dp[i - 1][j] * j + dp[i - 1][j - 1]) % MOD

        total_ways = 0
        for i in range(1, x + 1):
            total_ways = (total_ways + dp[n][i] * pow(y, i, MOD)) % MOD

        return total_ways