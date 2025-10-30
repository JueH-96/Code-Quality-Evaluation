class Solution:
  def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
    MOD = 10**9 + 7

    # dp[i][j][0] stores the number of stable arrays with i zeros and j ones, ending with 0.
    # dp[i][j][1] stores the number of stable arrays with i zeros and j ones, ending with 1.
    # Dimensions: (zero+1) x (one+1) x 2. dp[num_zeros][num_ones][ends_with_digit]
    dp = [[[0, 0] for _ in range(one + 1)] for _ in range(zero + 1)]

    # Base Cases:
    # Arrays consisting of only zeros: "0", "00", ... (up to 'limit' zeros)
    for i_zeros in range(1, zero + 1):
        if i_zeros <= limit:
            dp[i_zeros][0][0] = 1
        # else dp[i_zeros][0][0] remains 0, as it's an invalid block of zeros.
    
    # Arrays consisting of only ones: "1", "11", ... (up to 'limit' ones)
    for j_ones in range(1, one + 1):
        if j_ones <= limit:
            dp[0][j_ones][1] = 1
        # else dp[0][j_ones][1] remains 0, as it's an invalid block of ones.

    # Fill the DP table
    # Iterate i (current number of zeros) from 1 to zero.
    # Iterate j (current number of ones) from 1 to one.
    # Note: dp[0][j][0] and dp[i][0][1] are 0, which is correct.
    # dp[0][0][0] and dp[0][0][1] are also 0.
    
    for i in range(1, zero + 1):
        for j in range(1, one + 1):
            # Calculate dp[i][j][0] (arrays ending with 0)
            # This corresponds to S0(i,j) in the derivation notes.
            # S0(i,j) = sum_{k=1 to min(i, limit)} dp[i-k][j][1]
            #
            # The sum (dp[i-1][j][1] + dp[i-1][j][0]) represents:
            #   dp[i-1][j][1]: prefix ends in 1, append a single 0.
            #   dp[i-1][j][0]: prefix ends in 0, append another 0 (extending the block of 0s).
            # This sum correctly calculates sum_{k=1 to i} dp[i-k][j][1] if i <= limit,
            # or sum_{k=1 to limit} dp[i-k][j][1] if i > limit by removing the oldest term.
            
            # sum_prev_options_for_0 is S1(i-1,j) + S0(i-1,j)
            sum_prev_options_for_0 = (dp[i-1][j][1] + dp[i-1][j][0]) % MOD
            
            val_ends_0: int
            if i <= limit:
                val_ends_0 = sum_prev_options_for_0
            else: # i > limit
                # Term to subtract: dp[i-limit-1][j][1] (corresponds to S1(i-limit-1,j))
                # This removes prefixes that would result in a block of (limit+1) zeros.
                term_to_subtract_0 = dp[i-limit-1][j][1]
                val_ends_0 = (sum_prev_options_for_0 - term_to_subtract_0 + MOD) % MOD
            dp[i][j][0] = val_ends_0

            # Calculate dp[i][j][1] (arrays ending with 1)
            # Symmetric logic for S1(i,j).
            # S1(i,j) = sum_{k=1 to min(j, limit)} dp[i][j-k][0]
            
            # sum_prev_options_for_1 is S0(i,j-1) + S1(i,j-1)
            sum_prev_options_for_1 = (dp[i][j-1][0] + dp[i][j-1][1]) % MOD

            val_ends_1: int
            if j <= limit:
                val_ends_1 = sum_prev_options_for_1
            else: # j > limit
                # Term to subtract: dp[i][j-limit-1][0] (corresponds to S0(i,j-limit-1))
                term_to_subtract_1 = dp[i][j-limit-1][0]
                val_ends_1 = (sum_prev_options_for_1 - term_to_subtract_1 + MOD) % MOD
            dp[i][j][1] = val_ends_1
            
    # The total number of stable arrays is the sum of those ending in 0 and those ending in 1.
    ans = (dp[zero][one][0] + dp[zero][one][1]) % MOD
    return ans