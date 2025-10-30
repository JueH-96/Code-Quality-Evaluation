class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        n = zero + one
        dp = [[[0]*(limit + 1) for _ in range(one + 1)] for _ in range(zero + 1)]
        
        # Initialize base cases
        for i in range(zero + 1):
            for j in range(one + 1):
                dp[i][j][0] = 1
        
        # Fill up the dp table
        for k in range(1, n + 1):
            for i in range(zero, -1, -1):
                for j in range(one, -1, -1):
                    for last in range(1, limit + 1):
                        if i > 0:
                            dp[i][j][last] = (dp[i][j][last] + dp[i - 1][j][min(limit, last + 1)]) % MOD
                        if j > 0:
                            dp[i][j][last] = (dp[i][j][last] + dp[i][j - 1][1]) % MOD
        
        # The answer is stored in dp[zero][one][limit]
        return dp[zero][one][limit]