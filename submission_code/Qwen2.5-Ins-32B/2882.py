class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        dp = [0] * (n + 1)
        dp[0] = 1
        
        # Generate all x-th powers less than or equal to n
        powers = [i**x for i in range(1, n + 1) if i**x <= n]
        
        # Update dp array
        for power in powers:
            for j in range(n, power - 1, -1):
                dp[j] = (dp[j] + dp[j - power]) % MOD
        
        return dp[n]