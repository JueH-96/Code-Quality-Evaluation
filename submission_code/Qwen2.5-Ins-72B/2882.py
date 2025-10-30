class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        
        # Precompute the powers of numbers up to n
        powers = []
        i = 1
        while i**x <= n:
            powers.append(i**x)
            i += 1
        
        # Initialize the DP table
        dp = [0] * (n + 1)
        dp[0] = 1  # There's one way to make 0, which is using no numbers
        
        # Fill the DP table
        for power in powers:
            for j in range(n, power - 1, -1):
                dp[j] = (dp[j] + dp[j - power]) % MOD
        
        return dp[n]