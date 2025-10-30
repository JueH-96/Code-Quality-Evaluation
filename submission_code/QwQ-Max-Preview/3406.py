class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        # Initialize DP table with dimensions (zero+1) x (one+1) x 2 x (limit+1)
        dp = [[[[0] * (limit + 1) for _ in range(2)] for __ in range(one + 1)] for ___ in range(zero + 1)]
        
        # Initialize starting states
        if zero >= 1:
            dp[zero - 1][one][0][1] = 1
        if one >= 1:
            dp[zero][one - 1][1][1] = 1
        
        # Iterate from the maximum possible sum (zero + one - 1) down to 0
        for s in range(zero + one - 1, -1, -1):
            start_z = max(0, s - one)
            end_z = min(zero, s)
            for z in range(start_z, end_z + 1):
                o = s - z
                if o < 0 or o > one:
                    continue
                for d in [0, 1]:
                    for r in range(1, limit + 1):
                        current = dp[z][o][d][r]
                        if current == 0:
                            continue
                        # Try transitioning to both possible next digits
                        for next_d in [0, 1]:
                            if next_d == d:
                                # Same digit, check run length and availability
                                if d == 0:
                                    if z == 0 or r >= limit:
                                        continue
                                    new_z = z - 1
                                    new_o = o
                                    new_r = r + 1
                                    if new_z >= 0 and new_o >= 0 and new_r <= limit:
                                        dp[new_z][new_o][d][new_r] = (dp[new_z][new_o][d][new_r] + current) % MOD
                                else:  # d == 1
                                    if o == 0 or r >= limit:
                                        continue
                                    new_o = o - 1
                                    new_z = z
                                    new_r = r + 1
                                    if new_z >= 0 and new_o >= 0 and new_r <= limit:
                                        dp[new_z][new_o][d][new_r] = (dp[new_z][new_o][d][new_r] + current) % MOD
                            else:
                                # Different digit, check availability
                                if next_d == 0:
                                    if z == 0:
                                        continue
                                    new_z = z - 1
                                    new_o = o
                                    new_r = 1
                                    if new_z >= 0 and new_o >= 0:
                                        dp[new_z][new_o][0][new_r] = (dp[new_z][new_o][0][new_r] + current) % MOD
                                else:  # next_d == 1
                                    if o == 0:
                                        continue
                                    new_o = o - 1
                                    new_z = z
                                    new_r = 1
                                    if new_z >= 0 and new_o >= 0:
                                        dp[new_z][new_o][1][new_r] = (dp[new_z][new_o][1][new_r] + current) % MOD
        
        # Sum all valid ending states where zeros and ones are exhausted
        total = 0
        for d in [0, 1]:
            for r in range(1, limit + 1):
                total = (total + dp[0][0][d][r]) % MOD
        return total