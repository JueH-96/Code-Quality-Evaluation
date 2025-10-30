class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = 10**9 + 7

        # Precompute factorials and inverse factorials for combinations
        max_val = n + x
        fact = [1] * (max_val + 1)
        inv_fact = [1] * (max_val + 1)

        for i in range(2, max_val + 1):
            fact[i] = fact[i - 1] * i % MOD

        inv_fact[max_val] = pow(fact[max_val], MOD - 2, MOD)
        for i in range(max_val - 1, 0, -1):
            inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD

        def comb(n, k):
            if n < k:
                return 0
            return fact[n] * inv_fact[k] % MOD * inv_fact[n - k] % MOD

        # Dynamic programming to calculate the number of ways to distribute performers
        dp = [[0] * (x + 1) for _ in range(n + 1)]
        dp[0][0] = 1

        for i in range(1, n + 1):
            for j in range(1, x + 1):
                dp[i][j] = (dp[i][j] + dp[i - 1][j - 1]) % MOD
                dp[i][j] = (dp[i][j] + dp[i - 1][j] * j % MOD) % MOD

        total_ways = 0
        for j in range(1, x + 1):
            total_ways = (total_ways + dp[n][j] * pow(y, j, MOD)) % MOD

        return total_ways