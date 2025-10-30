class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        
        # Create a DP table
        dp = [0] * (n + 1)
        
        # Base case: There's one way to express 0 (by not selecting any integers)
        dp[0] = 1
        
        # For each integer i, update the DP table
        i = 1
        while i**x <= n:
            # Process in reverse order to ensure each integer is used at most once
            for j in range(n, i**x - 1, -1):
                dp[j] = (dp[j] + dp[j - i**x]) % MOD
            i += 1
        
        return dp[n]