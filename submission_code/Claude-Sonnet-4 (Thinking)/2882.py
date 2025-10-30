class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        
        # dp[i] = number of ways to express i as sum of x-th powers of unique positive integers
        dp = [0] * (n + 1)
        dp[0] = 1  # One way to express 0: use no integers
        
        # Iterate through all possible integers
        i = 1
        while i**x <= n:
            power = i**x
            # Update dp array in reverse order to avoid using the same integer multiple times
            for j in range(n, power - 1, -1):
                dp[j] = (dp[j] + dp[j - power]) % MOD
            i += 1
        
        return dp[n]