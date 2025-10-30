class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        dp = [[0] * (one + 1) for _ in range(zero + 1)]
        dp[0][0] = 1
        
        for i in range(zero + 1):
            for j in range(one + 1):
                if i > 0:
                    dp[i][j] = (dp[i][j] + dp[i - 1][j]) % MOD
                if j > 0:
                    dp[i][j] = (dp[i][j] + dp[i][j - 1]) % MOD
                if i > limit:
                    dp[i][j] = (dp[i][j] - dp[i - limit - 1][j] + MOD) % MOD
                if j > limit:
                    dp[i][j] = (dp[i][j] - dp[i][j - limit - 1] + MOD) % MOD
        
        return dp[zero][one]