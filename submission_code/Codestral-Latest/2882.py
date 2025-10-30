class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7

        # Precompute the x-th powers of integers up to n
        max_val = int(n ** (1/x)) + 1
        powers = [i**x for i in range(1, max_val + 1)]

        # DP array to store the number of ways to sum up to each value
        dp = [0] * (n + 1)
        dp[0] = 1  # There is one way to sum to 0 (using an empty set)

        for power in powers:
            for j in range(n, power - 1, -1):
                dp[j] = (dp[j] + dp[j - power]) % MOD

        return dp[n]