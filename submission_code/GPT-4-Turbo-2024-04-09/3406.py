class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        
        # Edge case where it's impossible to have a stable array
        if zero == 0 or one == 0:
            return 0
        
        # dp[i][j][k] will be the number of valid sequences of length i with j zeros and k ones
        # where the sequence does not end in more than `limit` consecutive same numbers
        dp = [[[0 for _ in range(one + 1)] for _ in range(zero + 1)] for _ in range(zero + one + 1)]
        
        # Initialization: starting with one zero or one one
        if zero > 0:
            dp[1][1][0] = 1
        if one > 0:
            dp[1][0][1] = 1
        
        # Fill the dp table
        for length in range(2, zero + one + 1):
            for z in range(zero + 1):
                for o in range(one + 1):
                    if z > 0:
                        # Add zeros to the sequence
                        # We can add a zero if the previous sequence has less than `limit` consecutive zeros ending
                        dp[length][z][o] += dp[length - 1][z - 1][o]
                        dp[length][z][o] %= MOD
                    if o > 0:
                        # Add ones to the sequence
                        # We can add a one if the previous sequence has less than `limit` consecutive ones ending
                        dp[length][z][o] += dp[length - 1][z][o - 1]
                        dp[length][z][o] %= MOD
        
        # We need the full length array with exactly 'zero' zeros and 'one' ones
        return dp[zero + one][zero][one]