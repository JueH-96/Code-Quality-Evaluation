class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 1000000007
        # Find the maximum number whose x-th power is <= n
        max_num = 0
        while (max_num + 1) ** x <= n:
            max_num += 1
        
        # Initialize dp table
        dp = [[0 for _ in range(n + 1)] for _ in range(max_num + 1)]
        dp[0][0] = 1
        
        # Fill the dp table
        for i in range(1, max_num + 1):
            pow_ix = i ** x
            for s in range(n + 1):
                dp[i][s] = (dp[i - 1][s] + (dp[i - 1][s - pow_ix] if s >= pow_ix else 0)) % MOD
        
        return dp[max_num][n]