from typing import List
import math

class Solution:
    def maxSum(self, nums: List[int], k: int, m: int) -> int:
        n = len(nums)

        # Precompute prefix sums
        # P[i] stores the sum of nums[0]...nums[i-1]. P[0] = 0.
        # This allows calculating sum of nums[a...b] as P[b+1] - P[a].
        P = [0] * (n + 1)
        for i in range(n):
            P[i+1] = P[i] + nums[i]

        # dp[j][i] represents the maximum sum of `j` non-overlapping subarrays
        # chosen from `nums[0...i-1]`, where each subarray has length at least `m`.
        # Initialize with -infinity to represent unreachable states.
        dp = [[-math.inf] * (n + 1) for _ in range(k + 1)]

        # Base case: 0 subarrays from any prefix always results in a sum of 0.
        for i in range(n + 1):
            dp[0][i] = 0

        # Iterate over the number of subarrays (j)
        for j in range(1, k + 1):
            # max_val_for_prev_j is a running maximum of (dp[j-1][p_prime] - P[p_prime])
            # for valid p_prime values. This is used to efficiently calculate the
            # maximum sum for the previous j-1 subarrays that allows the j-th subarray
            # to end at the current index 'i-1'.
            max_val_for_prev_j = -math.inf

            # Iterate over the end index (i) of the current prefix nums[0...i-1]
            for i in range(1, n + 1):
                # Option 1: The j-th subarray does not end at index i-1.
                # In this case, the max sum is the same as considering nums[0...i-2].
                dp[j][i] = dp[j][i-1]

                # Option 2: The j-th subarray ends at index i-1.
                # This j-th subarray is nums[p...i-1]. Its sum is P[i] - P[p].
                # The previous j-1 subarrays must be chosen from nums[0...p-1],
                # yielding a sum of dp[j-1][p].
                # Total sum: dp[j-1][p] + (P[i] - P[p]) = (dp[j-1][p] - P[p]) + P[i].
                # We need to maximize this over valid 'p'.
                # 'p' (the effective length of the prefix for j-1 subarrays)
                # must be between (j-1)*m (minimum for j-1 subarrays) and i-m (to allow current subarray of length m).
                
                # The condition `i >= j * m` ensures that 'i' is large enough to
                # accommodate `j` subarrays, each of length at least `m`.
                # This implicitly guarantees that `i - m >= (j - 1) * m`, which is
                # the minimum length required for `j-1` subarrays.
                if i >= j * m:
                    # Update max_val_for_prev_j by considering `p_prime = i-m`.
                    # `dp[j-1][i-m]` is the max sum for `j-1` subarrays from `nums[0...i-m-1]`.
                    if dp[j-1][i-m] != -math.inf:
                        max_val_for_prev_j = max(max_val_for_prev_j, dp[j-1][i-m] - P[i-m])

                    # If a valid previous state (max_val_for_prev_j) exists,
                    # we can form the j-th subarray ending at i-1.
                    if max_val_for_prev_j != -math.inf:
                        dp[j][i] = max(dp[j][i], max_val_for_prev_j + P[i])
        
        # The result is the maximum sum using k subarrays from the entire array nums.
        return dp[k][n]