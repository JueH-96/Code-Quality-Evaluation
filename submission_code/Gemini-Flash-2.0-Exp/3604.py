class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = 10**9 + 7
        
        # dp[i][j] represents the number of ways to assign i performers to j stages
        dp = [[0] * (x + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        
        for i in range(1, n + 1):
            for j in range(1, x + 1):
                dp[i][j] = (dp[i][j-1] + dp[i-1][j] * j) % MOD
        
        ways_to_assign = dp[n][x]
        
        return (ways_to_assign * pow(y, x, MOD)) % MOD