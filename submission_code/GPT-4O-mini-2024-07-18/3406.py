class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        
        # Dynamic programming table
        dp = [[0] * (one + 1) for _ in range(zero + 1)]
        dp[0][1] = 1  # Base case: one way to have one 1 and no 0s
        
        for z in range(zero + 1):
            for o in range(1, one + 1):
                # If we can add a 0
                if z > 0:
                    dp[z][o] = (dp[z][o] + dp[z - 1][o]) % MOD
                
                # If we can add a 1
                if o > 1:
                    dp[z][o] = (dp[z][o] + dp[z][o - 1]) % MOD
                
                # If we can add a 1 and a 0 together
                if z > 0 and o > 1:
                    dp[z][o] = (dp[z][o] + dp[z - 1][o - 1]) % MOD
                
                # Remove the invalid configurations
                if z > limit:
                    dp[z][o] = (dp[z][o] - dp[z - limit - 1][o] + MOD) % MOD
        
        return dp[zero][one]