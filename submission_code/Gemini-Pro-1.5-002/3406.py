class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        n = zero + one
        dp = [[0] * (one + 1) for _ in range(zero + 1)]
        dp[0][0] = 1
        ans = 0
        for i in range(zero + 1):
            for j in range(one + 1):
                if i < zero:
                    dp[i + 1][j] = (dp[i + 1][j] + dp[i][j]) % MOD
                if j < one:
                    dp[i][j + 1] = (dp[i][j + 1] + dp[i][j]) % MOD
                if i > 0 and j > 0 and (i > limit or j > limit):
                    if i > limit:
                        dp[i][j] = (dp[i][j] - dp[i - limit - 1][j] + MOD) % MOD
                    if j > limit:
                        dp[i][j] = (dp[i][j] - dp[i][j - limit - 1] + MOD) % MOD
                    if i > limit and j > limit:
                        dp[i][j] = (dp[i][j] + dp[i - limit - 1][j - limit - 1]) % MOD
                if i == zero and j == one:
                    ans = dp[i][j]
        return ans