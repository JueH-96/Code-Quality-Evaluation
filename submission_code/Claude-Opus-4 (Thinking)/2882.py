class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        
        # dp[j] = number of ways to get sum j
        dp = [0] * (n + 1)
        dp[0] = 1  # one way to get sum 0 (empty set)
        
        # Try each number i
        i = 1
        while i ** x <= n:
            power = i ** x
            # Traverse from right to left to avoid using the same number multiple times
            for j in range(n, power - 1, -1):
                dp[j] = (dp[j] + dp[j - power]) % MOD
            i += 1
        
        return dp[n]