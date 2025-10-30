class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        
        # Initialize DP table
        dp = [[[0 for _ in range(limit + 1)] for _ in range(one + 1)] for _ in range(zero + 1)]
        
        # Base case: one 0 and one 1
        dp[1][1][1] = 1
        
        # Fill the DP table
        for z in range(zero + 1):
            for o in range(one + 1):
                for l in range(1, limit + 1):
                    if z > 0:
                        dp[z][o][l] += dp[z - 1][o][max(1, l - 1)]
                        dp[z][o][l] %= MOD
                    if o > 0:
                        dp[z][o][l] += dp[z][o - 1][max(1, l - 1)]
                        dp[z][o][l] %= MOD
        
        # Sum up all valid arrays
        result = 0
        for l in range(1, limit + 1):
            result += dp[zero][one][l]
            result %= MOD
        
        return result