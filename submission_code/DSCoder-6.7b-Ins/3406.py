class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        dp = [[0]*(limit+2) for _ in range(zero+one+1)]
        dp[0][0] = 1
        for i in range(zero+one):
            for j in range(min(i+1, limit+1)):
                if j < limit:
                    dp[i+1][j+1] = (dp[i+1][j+1] + dp[i][j]) % MOD
                dp[i+1][j] = (dp[i+1][j] + dp[i][j]*(zero if j > 0 else one)) % MOD
        return sum(dp[-1]) % MOD