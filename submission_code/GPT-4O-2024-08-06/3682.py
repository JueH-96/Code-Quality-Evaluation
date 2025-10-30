class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        MOD = 10**9 + 7
        
        # dp[i][j] will store the number of ways to form an array of length i with j adjacent equal pairs
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        
        # Base case: there's only one way to form an array of length 1 with 0 adjacent equal pairs
        dp[1][0] = m
        
        for i in range(2, n + 1):
            for j in range(k + 1):
                # If the last two elements are different
                dp[i][j] = dp[i - 1][j] * (m - 1) % MOD
                # If the last two elements are the same
                if j > 0:
                    dp[i][j] = (dp[i][j] + dp[i - 1][j - 1]) % MOD
        
        return dp[n][k]