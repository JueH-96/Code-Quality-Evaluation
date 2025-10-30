class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        # Initialize a 4D DP array: dp[z][o][last][run]
        # last can be 0, 1, or 2 (where 2 represents no previous digit)
        dp = [[[[0] * (limit + 1) for _ in range(3)] for __ in range(one + 1)] for ___ in range(zero + 1)]
        dp[0][0][2][0] = 1  # Initial state: no digits used
        
        for z in range(zero + 1):
            for o in range(one + 1):
                for last in range(3):
                    for run in range(limit + 1):
                        current = dp[z][o][last][run]
                        if current == 0:
                            continue
                        if last == 2:
                            # Can add 0 or 1
                            if z < zero:
                                nz, no = z + 1, o
                                dp[nz][no][0][1] = (dp[nz][no][0][1] + current) % MOD
                            if o < one:
                                nz, no = z, o + 1
                                dp[nz][no][1][1] = (dp[nz][no][1][1] + current) % MOD
                        elif last == 0:
                            # Can add 0 if run+1 <= limit
                            if z < zero and run + 1 <= limit:
                                nz, no = z + 1, o
                                dp[nz][no][0][run + 1] = (dp[nz][no][0][run + 1] + current) % MOD
                            # Can add 1
                            if o < one:
                                nz, no = z, o + 1
                                dp[nz][no][1][1] = (dp[nz][no][1][1] + current) % MOD
                        elif last == 1:
                            # Can add 1 if run+1 <= limit
                            if o < one and run + 1 <= limit:
                                nz, no = z, o + 1
                                dp[nz][no][1][run + 1] = (dp[nz][no][1][run + 1] + current) % MOD
                            # Can add 0
                            if z < zero:
                                nz, no = z + 1, o
                                dp[nz][no][0][1] = (dp[nz][no][0][1] + current) % MOD
        
        result = 0
        # Sum all valid states at zero and one with last digit 0 or 1 and any valid run
        for last in [0, 1]:
            for run in range(1, limit + 1):
                result = (result + dp[zero][one][last][run]) % MOD
        return result