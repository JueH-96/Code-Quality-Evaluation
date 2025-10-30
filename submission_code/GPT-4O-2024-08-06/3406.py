class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        
        # Total length of the array
        total_length = zero + one
        
        # Initialize a dp array where dp[i][j] means the number of ways to arrange
        # i zeros and j ones such that no subarray of length > limit is homogeneous.
        dp = [[0] * (one + 1) for _ in range(zero + 1)]
        
        # Base case: one way to arrange 0 zeros and 0 ones
        dp[0][0] = 1
        
        # Fill the dp table
        for z in range(zero + 1):
            for o in range(one + 1):
                if z > 0:
                    # Add a zero if possible
                    dp[z][o] += dp[z - 1][o]
                    dp[z][o] %= MOD
                if o > 0:
                    # Add a one if possible
                    dp[z][o] += dp[z][o - 1]
                    dp[z][o] %= MOD
                
                # Check for invalid subarrays
                if z > limit:
                    dp[z][o] -= dp[z - limit - 1][o]
                    dp[z][o] %= MOD
                if o > limit:
                    dp[z][o] -= dp[z][o - limit - 1]
                    dp[z][o] %= MOD
        
        # The answer is the number of ways to arrange zero zeros and one ones
        return dp[zero][one]