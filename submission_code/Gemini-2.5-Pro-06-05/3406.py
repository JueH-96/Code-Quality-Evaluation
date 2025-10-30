class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7

        # dp[i][j][0]: number of stable arrays with i zeros and j ones, ending in 0.
        # dp[i][j][1]: number of stable arrays with i zeros and j ones, ending in 1.
        dp = [[[0] * 2 for _ in range(one + 1)] for _ in range(zero + 1)]

        # To optimize the calculation of sums, we use prefix sums.
        # dp_sum0[i][j]: sum(dp[i][k][0] for k in 0..j) (row-wise prefix sum)
        # dp_sum1[i][j]: sum(dp[k][j][1] for k in 0..i) (column-wise prefix sum)
        dp_sum0 = [[0] * (one + 1) for _ in range(zero + 1)]
        dp_sum1 = [[0] * (one + 1) for _ in range(zero + 1)]

        # Seed the DP for base cases. An "empty" string can be thought of as a
        # valid prefix. dp[0][0][k]=1 are computational aids.
        dp[0][0][0] = 1
        dp[0][0][1] = 1
        dp_sum0[0][0] = 1
        dp_sum1[0][0] = 1

        for i in range(zero + 1):
            for j in range(one + 1):
                if i == 0 and j == 0:
                    continue

                # Calculate dp[i][j][0]: arrays ending in 0.
                # This is sum(dp[i-c][j][1] for c=1..limit), which is optimized
                # using column-wise prefix sums of dp...[1].
                if i > 0:
                    sum_prev_1 = dp_sum1[i - 1][j]
                    if i - limit - 1 >= 0:
                        sum_prev_1 -= dp_sum1[i - limit - 1][j]
                    dp[i][j][0] = (sum_prev_1 + MOD) % MOD

                # Calculate dp[i][j][1]: arrays ending in 1.
                # This is sum(dp[i][j-c][0] for c=1..limit), optimized
                # using row-wise prefix sums of dp...[0].
                if j > 0:
                    sum_prev_0 = dp_sum0[i][j - 1]
                    if j - limit - 1 >= 0:
                        sum_prev_0 -= dp_sum0[i][j - limit - 1]
                    dp[i][j][1] = (sum_prev_0 + MOD) % MOD

                # Update prefix sums for the current (i, j) cell.
                dp_sum0[i][j] = ((dp_sum0[i][j - 1] if j > 0 else 0) + dp[i][j][0]) % MOD
                dp_sum1[i][j] = ((dp_sum1[i - 1][j] if i > 0 else 0) + dp[i][j][1]) % MOD

        # The result is the sum of arrays with `zero` 0s and `one` 1s,
        # ending in either 0 or 1.
        total_stable_arrays = (dp[zero][one][0] + dp[zero][one][1]) % MOD
        return total_stable_arrays