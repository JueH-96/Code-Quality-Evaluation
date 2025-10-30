class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        MOD = 10**9 + 7
        
        # Initialize dp array
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        
        # Base case: for n=1, we can choose any number from 1 to m
        for i in range(1, m + 1):
            dp[1][0] = m
        
        # Fill the dp table
        for i in range(2, n + 1):
            for j in range(min(k, i - 1) + 1):
                if j == 0:
                    # If no equal adjacent elements are required
                    dp[i][j] = (m * dp[i-1][j]) % MOD
                else:
                    # We can either choose the same number as previous (m choices)
                    # or choose a different number (m-1 choices)
                    dp[i][j] = (dp[i-1][j-1] + (m-1) * dp[i-1][j]) % MOD
        
        return dp[n][k]