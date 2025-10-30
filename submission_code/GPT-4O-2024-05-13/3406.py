class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        
        # dp[i][j] will store the number of valid arrays with i zeros and j ones
        dp = [[0] * (one + 1) for _ in range(zero + 1)]
        
        # Base case: one valid array with 0 zeros and 0 ones (empty array)
        dp[0][0] = 1
        
        for i in range(zero + 1):
            for j in range(one + 1):
                if i > 0:
                    for k in range(1, min(limit, i) + 1):
                        dp[i][j] = (dp[i][j] + dp[i - k][j]) % MOD
                if j > 0:
                    for k in range(1, min(limit, j) + 1):
                        dp[i][j] = (dp[i][j] + dp[i][j - k]) % MOD
        
        return dp[zero][one]