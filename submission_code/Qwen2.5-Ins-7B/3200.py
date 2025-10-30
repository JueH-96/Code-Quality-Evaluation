class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        dp = [[0] * 4 for _ in range(n + 1)]
        dp[0][0] = 1
        
        for i in range(1, n + 1):
            dp[i][0] = (dp[i-1][0] * 26) % MOD
            dp[i][1] = (dp[i-1][0] * 5) % MOD
            dp[i][2] = (dp[i-1][1] * 4) % MOD
            dp[i][3] = (dp[i-1][2] * 3) % MOD
        
        return (dp[n][0] + dp[n][1] + dp[n][2] + dp[n][3]) % MOD