from typing import List

class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        n = len(nums)
        # Compute the difference for the first element
        diff_val = target[0] - nums[0]
        # Initialize S with the absolute difference from 0 to D[0]
        S = abs(diff_val)
        # Set previous difference
        prev_diff = diff_val
        # Add the sum of absolute differences between consecutive D[i] for i=1 to n-1
        for i in range(1, n):
            current_diff = target[i] - nums[i]
            S += abs(current_diff - prev_diff)
            prev_diff = current_diff  # Update previous to current
        # Add the absolute difference from D[n-1] to 0
        S += abs(prev_diff)
        # The minimum operations is S divided by 2
        return S // 2