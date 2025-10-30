import math
from typing import List

class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)

        # Step 1: Calculate prefix sums
        # P[i] stores the sum of nums[0] to nums[i-1]
        # P[0] = 0
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i+1] = prefix_sum[i] + nums[i]

        # Step 2: Initialize min_P_for_remainder array
        # min_P_for_remainder[r] stores the minimum prefix_sum[idx] found so far
        # such that idx % k == r.
        # Initialize with infinity for all remainders except 0.
        # prefix_sum[0] is 0, and 0 % k == 0, so min_P_for_remainder[0] starts at 0.
        min_P_for_remainder = [math.inf] * k
        min_P_for_remainder[0] = 0

        # Step 3: Initialize max_subarray_sum
        # Since subarray sums can be negative, initialize with negative infinity.
        max_subarray_sum = -math.inf

        # Step 4: Iterate through prefix sums to find the maximum valid subarray sum
        # j iterates from 1 to n (inclusive), representing the end+1 index of the subarray
        for j in range(1, n + 1):
            current_prefix_sum = prefix_sum[j]
            current_remainder = j % k

            # If we have seen a prefix_sum[i] such that i % k == current_remainder and i < j,
            # then P[j] - min_P_for_remainder[current_remainder] is a candidate sum.
            # The length of this subarray (from nums[i] to nums[j-1]) is (j - i).
            # Since j % k == i % k, then (j - i) % k == 0.
            if min_P_for_remainder[current_remainder] != math.inf:
                candidate_sum = current_prefix_sum - min_P_for_remainder[current_remainder]
                max_subarray_sum = max(max_subarray_sum, candidate_sum)
            
            # Update the minimum prefix sum for the current remainder.
            # We update after calculating the candidate sum for the current `j`,
            # because `prefix_sum[j]` itself can be a `P[i]` for a future `j' > j`.
            min_P_for_remainder[current_remainder] = min(min_P_for_remainder[current_remainder], current_prefix_sum)

        return max_subarray_sum