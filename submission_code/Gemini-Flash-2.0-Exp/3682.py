class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        MOD = 10**9 + 7

        if k > n - 1:
            return 0

        # dp[i][j] = number of arrays of length i with j adjacent equal elements
        dp = [[0] * n for _ in range(n)]

        # Base case: array of length 1 has 0 adjacent equal elements
        dp[1][0] = m

        for i in range(2, n + 1):
            for j in range(i):
                if j == 0:
                    dp[i][j] = (dp[i - 1][j] * (m - 1)) % MOD
                else:
                    dp[i][j] = (dp[i - 1][j] * (m - 1) + dp[i - 1][j - 1]) % MOD

        return dp[n][k]