class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        
        # dp[i] = number of ways to make sum i
        dp = [0] * (n + 1)
        dp[0] = 1  # one way to make sum 0 (empty set)
        
        # Try each base number starting from 1
        base = 1
        while base ** x <= n:
            power = base ** x
            
            # Update dp array from right to left to avoid using updated values
            # in the same iteration
            for target in range(n, power - 1, -1):
                dp[target] = (dp[target] + dp[target - power]) % MOD
            
            base += 1
        
        return dp[n]