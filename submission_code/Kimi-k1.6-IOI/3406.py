class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        # Initialize a 4D DP array: dp[i][j][last][cnt]
        dp = [[[[0] * (limit + 2) for _ in range(2)] for __ in range(one + 1)] for ___ in range(zero + 1)]
        
        # Initial states
        if zero >= 1:
            dp[1][0][0][1] = 1
        if one >= 1:
            dp[0][1][1][1] = 1
        
        # Iterate through all possible i (number of zeros used) and j (number of ones used)
        for i in range(zero + 1):
            for j in range(one + 1):
                for last in [0, 1]:
                    for cnt in range(1, limit + 1):
                        current = dp[i][j][last][cnt]
                        if current == 0:
                            continue
                        
                        # Transition 1: Add the same element again if possible
                        if cnt < limit:
                            new_i = i + (1 if last == 0 else 0)
                            new_j = j + (1 if last == 1 else 0)
                            if new_i <= zero and new_j <= one:
                                dp[new_i][new_j][last][cnt + 1] = (dp[new_i][new_j][last][cnt + 1] + current) % MOD
                        
                        # Transition 2: Add the other element
                        new_last = 1 - last
                        new_i = i + (1 if new_last == 0 else 0)
                        new_j = j + (1 if new_last == 1 else 0)
                        if new_i <= zero and new_j <= one:
                            dp[new_i][new_j][new_last][1] = (dp[new_i][new_j][new_last][1] + current) % MOD
        
        # Sum all valid states where exactly zero zeros and one ones are used
        result = 0
        for last in [0, 1]:
            for cnt in range(1, limit + 1):
                result = (result + dp[zero][one][last][cnt]) % MOD
        
        return result