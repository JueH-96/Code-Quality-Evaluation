class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        
        # Base case: if limit is 1, there can only be one stable array
        if limit == 1:
            return 1 if zero == one else 0
        
        # Initialize dp array
        dp = [[0] * (one + 1) for _ in range(zero + 1)]
        dp[0][0] = 1
        
        # Calculate the number of ways to arrange zeros and ones
        for z in range(zero + 1):
            for o in range(one + 1):
                if z + o > 1:
                    # Add zeros
                    if z > 0:
                        dp[z][o] += dp[z - 1][o] * (z if o < limit else z - 1)
                    # Add ones
                    if o > 0:
                        dp[z][o] += dp[z][o - 1] * (o if z < limit else o - 1)
                    dp[z][o] %= MOD
        
        return dp[zero][one]