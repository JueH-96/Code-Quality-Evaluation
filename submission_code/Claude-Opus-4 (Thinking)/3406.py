class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        
        # dp[i][j][end][length] = number of ways to arrange i 0s and j 1s
        # where the array ends with 'end' (0 or 1) and the last segment has 'length' consecutive elements
        dp = [[[[0 for _ in range(limit + 1)] for _ in range(2)] for _ in range(one + 1)] for _ in range(zero + 1)]
        
        # Base cases
        dp[1][0][0][1] = 1  # Array is just [0]
        dp[0][1][1][1] = 1  # Array is just [1]
        
        # Fill the DP table
        for i in range(zero + 1):
            for j in range(one + 1):
                for end in range(2):
                    for length in range(1, limit + 1):
                        if dp[i][j][end][length] == 0:
                            continue
                        
                        current_ways = dp[i][j][end][length]
                        
                        # Try to add a 0
                        if i < zero:
                            if end == 0 and length < limit:
                                # Extend the current segment of 0s
                                dp[i+1][j][0][length+1] = (dp[i+1][j][0][length+1] + current_ways) % MOD
                            elif end == 1:
                                # Start a new segment of 0s
                                dp[i+1][j][0][1] = (dp[i+1][j][0][1] + current_ways) % MOD
                        
                        # Try to add a 1
                        if j < one:
                            if end == 1 and length < limit:
                                # Extend the current segment of 1s
                                dp[i][j+1][1][length+1] = (dp[i][j+1][1][length+1] + current_ways) % MOD
                            elif end == 0:
                                # Start a new segment of 1s
                                dp[i][j+1][1][1] = (dp[i][j+1][1][1] + current_ways) % MOD
        
        # Sum up all valid final states
        result = 0
        for end in range(2):
            for length in range(1, limit + 1):
                result = (result + dp[zero][one][end][length]) % MOD
        
        return result