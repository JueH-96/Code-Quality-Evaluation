class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        
        # Precompute powers of integers up to n
        powers = []
        i = 1
        while (power := i ** x) <= n:
            powers.append(power)
            i += 1
        
        # Dynamic programming table
        dp = [0] * (n + 1)
        dp[0] = 1  # There's one way to sum to 0: using no numbers
        
        # Fill the dp table
        for power in powers:
            for j in range(n, power - 1, -1):
                dp[j] = (dp[j] + dp[j - power]) % MOD
        
        return dp[n]