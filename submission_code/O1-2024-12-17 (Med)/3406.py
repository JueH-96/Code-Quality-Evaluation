class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        """
        We want to count the number of binary arrays of length (zero + one) that:
          1) Contain exactly 'zero' zeroes and 'one' ones.
          2) Every subarray of length > limit contains both 0 and 1.
        
        Condition (2) is equivalent to saying we cannot have a run of more than 'limit' identical bits.
        (Because if there were a run of length > limit, that run itself would be a subarray without both 0 and 1.)
        
        Hence, we just need to count how many ways to place exactly 'zero' zeroes and 'one' ones
        in a sequence so that no run of 0s or 1s exceeds 'limit'.
        
        We use DP where dp[i][j][last][run] = number of ways to form a sequence using i zeroes and j ones
        that ends with 'last' (0 or 1) repeated 'run' times consecutively (run <= limit).
        
        We start with an empty sequence (dp[0][0][0][0] = 1 as a "no last digit" placeholder)
        and build up, ensuring runs do not exceed 'limit'.
        """
        MOD = 10**9 + 7
        
        # dp[i][j][last][run] = number of ways to build a sequence with
        # i zeroes used, j ones used, ending with the bit 'last' repeated 'run' times.
        # run=0 is a special "empty sequence" placeholder (no bits yet).
        dp = [[[[0]*(limit+1) for _ in range(2)] for _ in range(one+1)] for _ in range(zero+1)]
        
        # Start with an empty sequence
        dp[0][0][0][0] = 1  # last=0, run=0 means "no bits chosen yet"
        
        for i in range(zero+1):
            for j in range(one+1):
                for last in (0, 1):
                    for run in range(limit+1):
                        count = dp[i][j][last][run]
                        if count == 0:
                            continue
                        
                        # If run=0, it means we have not placed any bit yet.
                        # We can start with a '0' or a '1' (if available).
                        if run == 0:
                            if i < zero:
                                dp[i+1][j][0][1] = (dp[i+1][j][0][1] + count) % MOD
                            if j < one:
                                dp[i][j+1][1][1] = (dp[i][j+1][1][1] + count) % MOD
                            continue
                        
                        # If we already have a last bit:
                        if last == 0:
                            # We can add another '0' if we still have zeroes left and won't exceed run limit
                            if i < zero and run < limit:
                                dp[i+1][j][0][run+1] = (dp[i+1][j][0][run+1] + count) % MOD
                            # We can add a '1' if we still have ones left
                            if j < one:
                                dp[i][j+1][1][1] = (dp[i][j+1][1][1] + count) % MOD
                        else:  # last == 1
                            # We can add another '1' if we still have ones left and won't exceed run limit
                            if j < one and run < limit:
                                dp[i][j+1][1][run+1] = (dp[i][j+1][1][run+1] + count) % MOD
                            # We can add a '0' if we still have zeroes left
                            if i < zero:
                                dp[i+1][j][0][1] = (dp[i+1][j][0][1] + count) % MOD
        
        # Sum up all valid ways where we used exactly 'zero' zeroes and 'one' ones
        ans = 0
        for last in (0, 1):
            for run in range(1, limit+1):
                ans = (ans + dp[zero][one][last][run]) % MOD
        
        return ans