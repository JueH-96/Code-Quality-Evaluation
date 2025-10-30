class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        M = 10**9 + 7

        # dp[i][j][0]: number of stable arrays with i zeros, j ones, ending with 0
        # dp[i][j][1]: number of stable arrays with i zeros, j ones, ending with 1
        dp = [[[0] * 2 for _ in range(one + 1)] for _ in range(zero + 1)]

        # SumEnding0[i][j]: sum of dp[i][k][0] for k from 0 to j
        SumEnding0 = [[0] * (one + 1) for _ in range(zero + 1)]

        # SumEnding1[i][j]: sum of dp[k][j][1] for k from 0 to i
        SumEnding1 = [[0] * (one + 1) for _ in range(zero + 1)]

        # Base cases and DP transitions
        for i in range(zero + 1):
            for j in range(one + 1):
                # Calculate dp[i][j][0] (ends with 0)
                if i > 0:
                    if j == 0: # Array contains only 0s
                        if i <= limit:
                            dp[i][0][0] = 1
                    else: # Array ends with 0, previous must end with 1
                        # We append a block of k zeros (1 <= k <= min(i, limit))
                        # to a state (i-k, j) ending with 1.
                        # dp[i][j][0] = sum(dp[i-k][j][1] for k from 1 to min(i, limit))
                        # Let m = i-k. Sum is dp[m][j][1] for m from i-min(i, limit) to i-1.
                        
                        min_prev_i = i - min(i, limit) # The smallest i' = i-k

                        # Sum dp[m][j][1] for m from min_prev_i to i-1
                        sum_val = SumEnding1[i-1][j] # Sum dp[m][j][1] for m from 0 to i-1
                        if min_prev_i > 0:
                            # Subtract sum dp[m][j][1] for m from 0 to min_prev_i - 1
                            sum_val = (sum_val - SumEnding1[min_prev_i - 1][j] + M) % M
                        # If min_prev_i is 0, sum is from m=0, which is just SumEnding1[i-1][j].
                        
                        dp[i][j][0] = sum_val

                # Calculate dp[i][j][1] (ends with 1)
                if j > 0:
                    if i == 0: # Array contains only 1s
                        if j <= limit:
                            dp[0][j][1] = 1
                    else: # Array ends with 1, previous must end with 0
                        # We append a block of k ones (1 <= k <= min(j, limit))
                        # to a state (i, j-k) ending with 0.
                        # dp[i][j][1] = sum(dp[i][j-k][0] for k from 1 to min(j, limit))
                        # Let m = j-k. Sum is dp[i][m][0] for m from j-min(j, limit) to j-1.
                        
                        min_prev_j = j - min(j, limit) # The smallest j' = j-k

                        # Sum dp[i][m][0] for m from min_prev_j to j-1
                        sum_val = SumEnding0[i][j-1] # Sum dp[i][m][0] for m from 0 to j-1
                        if min_prev_j > 0:
                             # Subtract sum dp[i][m][0] for m from 0 to min_prev_j - 1
                            sum_val = (sum_val - SumEnding0[i][min_prev_j - 1] + M) % M
                        # If min_prev_j is 0, sum is from m=0, which is just SumEnding0[i][j-1].

                        dp[i][j][1] = sum_val

                # Update prefix sums for the current (i, j) state
                # SumEnding0[i][j] = sum(dp[i][k][0] for k=0..j)
                if j > 0:
                    SumEnding0[i][j] = (SumEnding0[i][j-1] + dp[i][j][0]) % M
                else: # j == 0
                    SumEnding0[i][0] = dp[i][0][0]

                # SumEnding1[i][j] = sum(dp[k][j][1] for k=0..i)
                if i > 0:
                    SumEnding1[i][j] = (SumEnding1[i-1][j] + dp[i][j][1]) % M
                else: # i == 0
                    SumEnding1[0][j] = dp[0][j][1]

        # The total number of stable arrays is the sum of arrays ending in 0 and ending in 1
        # with the exact total number of zeros and ones.
        return (dp[zero][one][0] + dp[zero][one][1]) % M