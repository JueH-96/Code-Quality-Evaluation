class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        # Initialize DP table: dp[z][o][last][run]
        dp = [[[[0] * (limit + 1) for _ in range(2)] for __ in range(one + 1)] for ___ in range(zero + 1)]
        
        # Initial conditions
        if zero >= 1:
            dp[1][0][0][1] = 1
        if one >= 1:
            dp[0][1][1][1] = 1
        
        # Iterate over all possible totals (sum of zeros and ones used so far)
        for total in range(1, zero + one):
            for z in range(zero + 1):
                o = total - z
                if o < 0 or o > one:
                    continue
                for last in [0, 1]:
                    for run in range(1, limit + 1):
                        current = dp[z][o][last][run]
                        if current == 0:
                            continue
                        # Try adding a 0
                        if z < zero:
                            new_z = z + 1
                            new_o = o
                            new_last = 0
                            if last == new_last:
                                new_run = run + 1
                                if new_run <= limit:
                                    dp[new_z][new_o][new_last][new_run] = (
                                        dp[new_z][new_o][new_last][new_run] + current
                                    ) % MOD
                            else:
                                new_run = 1
                                dp[new_z][new_o][new_last][new_run] = (
                                    dp[new_z][new_o][new_last][new_run] + current
                                ) % MOD
                        # Try adding a 1
                        if o < one:
                            new_z = z
                            new_o = o + 1
                            new_last = 1
                            if last == new_last:
                                new_run = run + 1
                                if new_run <= limit:
                                    dp[new_z][new_o][new_last][new_run] = (
                                        dp[new_z][new_o][new_last][new_run] + current
                                    ) % MOD
                            else:
                                new_run = 1
                                dp[new_z][new_o][new_last][new_run] = (
                                    dp[new_z][new_o][new_last][new_run] + current
                                ) % MOD
        
        # Sum all possibilities for the final state (zero, one)
        ans = 0
        for last in [0, 1]:
            for run in range(1, limit + 1):
                ans = (ans + dp[zero][one][last][run]) % MOD
        return ans