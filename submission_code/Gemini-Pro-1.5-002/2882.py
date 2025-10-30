class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        dp = [0] * (n + 1)
        dp[0] = 1
        
        for num in range(1, int(n**(1/x)) + 1):
            val = num**x
            for i in range(n, val - 1, -1):
                dp[i] = (dp[i] + dp[i - val]) % MOD
        
        return dp[n]