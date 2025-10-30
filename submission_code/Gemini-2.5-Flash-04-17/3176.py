from typing import List
import math

class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        n = len(nums)
        # Initialize minimum sum to infinity.
        # We use infinity so that the first valid mountain triplet sum
        # will always be smaller and update min_sum.
        min_sum = math.inf

        # Iterate through all possible triplets of indices (i, j, k)
        # such that 0 <= i < j < k < n.
        # The outer loop iterates possible index for 'i'.
        # 'i' must be at least 0, and at most n-3 (to ensure there are at least 2 elements after it for 'j' and 'k').
        for i in range(n - 2):
            # The middle loop iterates possible index for 'j'.
            # 'j' must be greater than 'i' (start from i + 1), and at most n-2 (to ensure there is at least 1 element after it for 'k').
            for j in range(i + 1, n - 1):
                # The inner loop iterates possible index for 'k'.
                # 'k' must be greater than 'j' (start from j + 1), and at most n-1 (the last index).
                for k in range(j + 1, n):
                    # Check if the triplet forms a mountain shape: nums[i] < nums[j] and nums[k] < nums[j].
                    if nums[i] < nums[j] and nums[k] < nums[j]:
                        # If it is a mountain triplet, calculate its sum.
                        current_sum = nums[i] + nums[j] + nums[k]
                        # Update the minimum sum found so far.
                        min_sum = min(min_sum, current_sum)

        # After checking all possible triplets, if min_sum is still infinity,
        # it means no mountain triplet was found in the array.
        if min_sum == math.inf:
            return -1
        else:
            # Otherwise, return the minimum sum found.
            return min_sum