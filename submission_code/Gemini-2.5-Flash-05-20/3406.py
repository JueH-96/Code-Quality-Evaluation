class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7

        # dp[i][j][type][k]
        # i: number of zeros used
        # j: number of ones used
        # type: 0 for ending with 0, 1 for ending with 1
        # k: number of consecutive 'type' elements at the end (1 <= k <= limit)
        dp = [[[[0] * (limit + 1) for _ in range(2)] for _ in range(one + 1)] for _ in range(zero + 1)]

        # sum_dp[i][j][type] stores sum of dp[i][j][type][k] for k from 1 to limit
        sum_dp = [[[0] * 2 for _ in range(one + 1)] for _ in range(zero + 1)]

        # Base cases: Arrays with a single element
        # Array: [0]
        if zero >= 1:
            dp[1][0][0][1] = 1
            sum_dp[1][0][0] = 1
        # Array: [1]
        if one >= 1:
            dp[0][1][1][1] = 1
            sum_dp[0][1][1] = 1

        # Iterate through all possible counts of zeros (i) and ones (j)
        for i in range(zero + 1):
            for j in range(one + 1):
                # Skip the (0,0) case as it's not a valid array, and skip base cases directly handled
                if i == 0 and j == 0:
                    continue
                # If current (i,j) corresponds to a base case like [0] or [1], these are already initialized.
                # The subsequent logic will correctly add to them if there are other paths to reach them
                # (which there aren't for single element arrays, only for extending them).

                # Calculate ways to end with 0
                if i > 0: # Check if we can add a 0
                    # Option 1: Extend a run of zeros. (e.g., ...00 from ...0)
                    # New sequence ends with k_zeros consecutive 0s. The previous was k_zeros-1 0s.
                    for k_zeros in range(2, limit + 1): # k_zeros must be at least 2 for extension
                        if i - 1 >= 0: # Ensure valid index for previous count of zeros
                            dp[i][j][0][k_zeros] = (dp[i][j][0][k_zeros] + dp[i-1][j][0][k_zeros-1]) % MOD
                    
                    # Option 2: Start a new run of zeros (add 0 after a 1). (e.g., ...10)
                    # New sequence ends with 1 consecutive 0.
                    # This comes from any sequence using (i-1) zeros and j ones, ending in any number of 1s.
                    if i - 1 >= 0: # Ensure valid index for previous count of zeros
                        dp[i][j][0][1] = (dp[i][j][0][1] + sum_dp[i-1][j][1]) % MOD

                # Calculate ways to end with 1
                if j > 0: # Check if we can add a 1
                    # Option 1: Extend a run of ones. (e.g., ...11 from ...1)
                    for k_ones in range(2, limit + 1):
                        if j - 1 >= 0: # Ensure valid index for previous count of ones
                            dp[i][j][1][k_ones] = (dp[i][j][1][k_ones] + dp[i][j-1][1][k_ones-1]) % MOD
                    
                    # Option 2: Start a new run of ones (add 1 after a 0). (e.g., ...01)
                    # New sequence ends with 1 consecutive 1.
                    # This comes from any sequence using i zeros and (j-1) ones, ending in any number of 0s.
                    if j - 1 >= 0: # Ensure valid index for previous count of ones
                        dp[i][j][1][1] = (dp[i][j][1][1] + sum_dp[i][j-1][0]) % MOD
                
                # After calculating all dp[i][j][type][k] for current (i,j), update sum_dp[i][j][type]
                # sum_dp for type 0 (ending with 0)
                current_sum_0 = 0
                for k in range(1, limit + 1):
                    current_sum_0 = (current_sum_0 + dp[i][j][0][k]) % MOD
                sum_dp[i][j][0] = current_sum_0

                # sum_dp for type 1 (ending with 1)
                current_sum_1 = 0
                for k in range(1, limit + 1):
                    current_sum_1 = (current_sum_1 + dp[i][j][1][k]) % MOD
                sum_dp[i][j][1] = current_sum_1

        # The total number of stable arrays is the sum of ways to form an array
        # with 'zero' 0s and 'one' 1s, ending in either 0 or 1.
        total_stable_arrays = (sum_dp[zero][one][0] + sum_dp[zero][one][1]) % MOD
        
        return total_stable_arrays