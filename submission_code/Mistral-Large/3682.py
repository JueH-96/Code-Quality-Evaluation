class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        MOD = 10**9 + 7

        # Initialize the DP table
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        dp[1][0] = m  # There are m ways to choose the first element

        # Fill the DP table
        for i in range(2, n + 1):
            for j in range(k + 1):
                if j > 0:
                    dp[i][j] = (dp[i][j] + dp[i - 1][j - 1]) % MOD
                if j < k:
                    dp[i][j] = (dp[i][j] + dp[i - 1][j] * (m - 1)) % MOD

        # The result is the sum of all dp[n][j] for j in range(k + 1)
        result = sum(dp[n]) % MOD
        return result