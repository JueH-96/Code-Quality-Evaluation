class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        mod = 10**9 + 7
        max_s = zero + one
        zero_plus1 = zero + 1
        one_plus1 = one + 1
        limit_plus1 = limit + 1
        
        # Initialize DP table with all zeros
        dp = [[[[0] * limit_plus1 for _ in range(2)] for __ in range(one_plus1)] for ___ in range(zero_plus1)]
        
        # Set initial states
        if zero > 0:
            dp[1][0][0][1] = 1
        if one > 0:
            dp[0][1][1][1] = 1
        
        # Process each possible sum s
        for s in range(1, max_s):
            for z in range(zero_plus1):
                o = s - z
                if o < 0 or o >= one_plus1:
                    continue
                for last in [0, 1]:
                    for l in range(1, limit_plus1):
                        cnt = dp[z][o][last][l]
                        if cnt == 0:
                            continue
                        # Transition to the same element
                        if last == 0:
                            new_z, new_o = z + 1, o
                            if new_z <= zero and l + 1 <= limit:
                                dp[new_z][new_o][0][l + 1] = (dp[new_z][new_o][0][l + 1] + cnt) % mod
                        else:
                            new_z, new_o = z, o + 1
                            if new_o <= one and l + 1 <= limit:
                                dp[new_z][new_o][1][l + 1] = (dp[new_z][new_o][1][l + 1] + cnt) % mod
                        # Transition to the other element
                        other = 1 - last
                        if other == 0:
                            new_z, new_o = z + 1, o
                            if new_z <= zero:
                                dp[new_z][new_o][0][1] = (dp[new_z][new_o][0][1] + cnt) % mod
                        else:
                            new_z, new_o = z, o + 1
                            if new_o <= one:
                                dp[new_z][new_o][1][1] = (dp[new_z][new_o][1][1] + cnt) % mod
        
        # Sum all valid configurations
        total = 0
        for last in [0, 1]:
            for l in range(1, limit_plus1):
                total = (total + dp[zero][one][last][l]) % mod
        return total