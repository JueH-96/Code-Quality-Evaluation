import math

class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = 1000000007
        # Initialize DP table
        dp = [[0 for _ in range(x + 1)] for _ in range(n + 1)]
        dp[0][0] = 1
        
        # Fill DP table
        for i in range(1, n + 1):
            dp[i][0] = 0  # No way to have 0 occupied stages with performers
            for s in range(1, min(i, x) + 1):
                val1 = (dp[i - 1][s] * s % MOD)
                val2 = (dp[i - 1][s - 1] * ((x - s + 1) % MOD) % MOD)
                dp[i][s] = (val1 + val2) % MOD
        
        # Compute the sum with y^s
        min_n_x = min(n, x)
        ans = 0
        current_pow = 1  # Start with y^0
        for s in range(0, min_n_x + 1):
            prod = (dp[n][s] * current_pow % MOD)
            ans = (ans + prod) % MOD
            # Update current_pow for the next power of y
            current_pow = (current_pow * (y % MOD) % MOD)
        
        return ans % MOD