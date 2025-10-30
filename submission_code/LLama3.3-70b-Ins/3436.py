from typing import List

class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        min_diff = float('inf')
        
        # Generate all possible subarrays
        for i in range(len(nums)):
            or_val = 0
            for j in range(i, len(nums)):
                # Calculate the bitwise OR of the current subarray
                or_val |= nums[j]
                # Calculate the absolute difference between k and the bitwise OR
                diff = abs(k - or_val)
                # Update the minimum difference
                min_diff = min(min_diff, diff)
        
        return min_diff