class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        
        # dp[i] will store the number of ways to express i using xth powers of unique integers
        dp = [0] * (n + 1)
        dp[0] = 1  # There's one way to sum up to 0: using an empty set
        
        # Calculate the maximum value that can be used as a base of x^th power
        max_base = int(n**(1/x))
        
        # Iterate over all possible bases that can be used
        for base in range(1, max_base + 1):
            current_power = base**x
            # Update dp table in reverse to avoid using the same base multiple times
            for j in range(n, current_power - 1, -1):
                dp[j] = (dp[j] + dp[j - current_power]) % MOD
        
        return dp[n]