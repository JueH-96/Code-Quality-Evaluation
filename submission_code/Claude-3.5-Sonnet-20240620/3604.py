class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = 10**9 + 7
        
        # Initialize the DP table
        dp = [[0] * (x + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        
        # Fill the DP table
        for i in range(1, n + 1):
            for j in range(1, x + 1):
                # Ways to assign i performers to j stages
                dp[i][j] = (dp[i-1][j-1] + dp[i-1][j] * j) % MOD
        
        # Calculate total ways
        total_ways = 0
        for j in range(1, x + 1):
            # For each possible number of non-empty stages,
            # multiply by the number of ways to assign scores
            total_ways = (total_ways + dp[n][j] * pow(y, j, MOD)) % MOD
        
        return total_ways