class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        
        # dp[i][j][k] = number of stable arrays using i zeros and j ones, 
        # ending with k (0 or 1)
        dp = [[[0] * 2 for _ in range(one + 1)] for _ in range(zero + 1)]
        
        # Base case
        dp[0][0][0] = dp[0][0][1] = 1
        
        for i in range(zero + 1):
            for j in range(one + 1):
                if i == 0 and j == 0:
                    continue
                
                # Calculate dp[i][j][0] (ending with 0)
                if i > 0:
                    # Sum over all possible lengths of the last run of 0s
                    for run_len in range(1, min(i, limit) + 1):
                        if i - run_len == 0 and j == 0:
                            # Special case: entire array is 0s
                            if run_len <= limit:
                                dp[i][j][0] = (dp[i][j][0] + 1) % MOD
                        else:
                            # Previous element must be 1
                            dp[i][j][0] = (dp[i][j][0] + dp[i - run_len][j][1]) % MOD
                
                # Calculate dp[i][j][1] (ending with 1)
                if j > 0:
                    # Sum over all possible lengths of the last run of 1s
                    for run_len in range(1, min(j, limit) + 1):
                        if i == 0 and j - run_len == 0:
                            # Special case: entire array is 1s
                            if run_len <= limit:
                                dp[i][j][1] = (dp[i][j][1] + 1) % MOD
                        else:
                            # Previous element must be 0
                            dp[i][j][1] = (dp[i][j][1] + dp[i][j - run_len][0]) % MOD
        
        return (dp[zero][one][0] + dp[zero][one][1]) % MOD