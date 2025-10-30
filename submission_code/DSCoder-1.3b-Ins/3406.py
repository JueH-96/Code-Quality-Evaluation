class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        dp = [[0]*(limit+1) for _ in range(zero+1)]
        dp[0][0] = 1
        for i in range(1, zero+1):
            for j in range(1, limit+1):
                dp[i][j] = dp[i-1][j] * (one+i-1)
                if j >= i:
                    dp[i][j] += dp[i][j-i] * (zero+i-1)
                dp[i][j] %= MOD
        return sum(dp[zero]) % MOD