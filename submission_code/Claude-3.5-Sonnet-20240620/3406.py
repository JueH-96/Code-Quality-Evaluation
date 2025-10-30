class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        
        # Initialize the DP table
        dp = [[[0 for _ in range(limit + 1)] for _ in range(one + 1)] for _ in range(zero + 1)]
        
        # Base case: when we have used all 0s and 1s
        dp[0][0][0] = 1
        
        # Fill the DP table
        for z in range(zero + 1):
            for o in range(one + 1):
                for l in range(limit + 1):
                    if z > 0:
                        dp[z][o][min(l + 1, limit)] = (dp[z][o][min(l + 1, limit)] + dp[z-1][o][l]) % MOD
                    if o > 0:
                        dp[z][o][1] = (dp[z][o][1] + dp[z][o-1][l]) % MOD
        
        # Sum up all valid configurations
        return sum(dp[zero][one]) % MOD