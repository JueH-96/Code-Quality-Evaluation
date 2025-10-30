class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        # Initialize dp arrays
        dp0 = [[[0 for _ in range(limit + 1)] for _ in range(one + 1)] for _ in range(zero + 1)]
        dp1 = [[[0 for _ in range(limit + 1)] for _ in range(one + 1)] for _ in range(zero + 1)]
        
        # Base cases
        if zero >= 1:
            dp0[1][0][1] = 1
        if one >= 1:
            dp1[0][1][1] = 1
        
        # Fill dp tables
        for i in range(zero + 1):
            for j in range(one + 1):
                for l in range(1, limit + 1):
                    # Transitions from dp0[i][j][l]
                    if dp0[i][j][l] > 0:
                        # Add another 0
                        if l + 1 <= limit and i + 1 <= zero:
                            dp0[i + 1][j][l + 1] = (dp0[i + 1][j][l + 1] + dp0[i][j][l]) % MOD
                        # Add a 1
                        if j + 1 <= one:
                            dp1[i][j + 1][1] = (dp1[i][j + 1][1] + dp0[i][j][l]) % MOD
                    # Transitions from dp1[i][j][l]
                    if dp1[i][j][l] > 0:
                        # Add another 1
                        if l + 1 <= limit and j + 1 <= one:
                            dp1[i][j + 1][l + 1] = (dp1[i][j + 1][l + 1] + dp1[i][j][l]) % MOD
                        # Add a 0
                        if i + 1 <= zero:
                            dp0[i + 1][j][1] = (dp0[i + 1][j][1] + dp1[i][j][l]) % MOD
        
        # Sum the results for dp0[zero][one][l] and dp1[zero][one][l] for l from 1 to limit
        result = 0
        for l in range(1, limit + 1):
            result = (result + dp0[zero][one][l]) % MOD
            result = (result + dp1[zero][one][l]) % MOD
        return result