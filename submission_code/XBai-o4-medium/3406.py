class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        
        # Initialize the 4D DP array
        dp = [[[[0]*(limit + 1) for _ in range(2)] for __ in range(one + 1)] for ___ in range(zero + 1)]
        
        # Base cases: first element is 0 or 1
        dp[1][0][0][1] = 1
        dp[0][1][1][1] = 1
        
        # Process all possible sums s from 1 to (zero + one)
        for s in range(1, zero + one + 1):
            for z in range(0, zero + 1):
                o_val = s - z
                if o_val < 0 or o_val > one:
                    continue
                # Iterate over last type (0 or 1) and run length
                for last in [0, 1]:
                    for run in range(1, limit + 1):
                        current = dp[z][o_val][last][run]
                        if current == 0:
                            continue
                        # Process transitions
                        if last == 0:
                            # Adding another 0
                            new_z = z + 1
                            new_o = o_val
                            new_run = run + 1
                            if new_z <= zero and new_run <= limit:
                                dp[new_z][new_o][0][new_run] = (dp[new_z][new_o][0][new_run] + current) % MOD
                            # Adding 1
                            new_o_add = o_val + 1
                            if new_o_add <= one:
                                dp[z][new_o_add][1][1] = (dp[z][new_o_add][1][1] + current) % MOD
                        else:  # last == 1
                            # Adding another 1
                            new_o = o_val + 1
                            new_run = run + 1
                            if new_o <= one and new_run <= limit:
                                dp[z][new_o][1][new_run] = (dp[z][new_o][1][new_run] + current) % MOD
                            # Adding 0
                            new_z_add = z + 1
                            if new_z_add <= zero:
                                dp[new_z_add][o_val][0][1] = (dp[new_z_add][o_val][0][1] + current) % MOD
        
        # Sum all valid states that use exactly 'zero' zeros and 'one' ones
        result = 0
        for last in [0, 1]:
            for run in range(1, limit + 1):
                result = (result + dp[zero][one][last][run]) % MOD
        
        return result