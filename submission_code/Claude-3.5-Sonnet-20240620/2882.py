class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        
        # Create a list of x-th powers up to n
        powers = [i**x for i in range(1, int(n**(1/x)) + 1)]
        
        # Initialize dp array
        dp = [0] * (n + 1)
        dp[0] = 1
        
        # Dynamic programming
        for power in powers:
            for i in range(n, power - 1, -1):
                dp[i] = (dp[i] + dp[i - power]) % MOD
        
        return dp[n]