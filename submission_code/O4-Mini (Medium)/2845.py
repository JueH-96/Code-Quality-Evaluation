from typing import List

class Solution:
    def findValueOfPartition(self, nums: List[int]) -> int:
        """
        We need to partition nums into two non-empty arrays such that
        |max(nums1) - min(nums2)| is minimized. Observing that the best
        such partition will split the sorted array between two adjacent
        elements, the problem reduces to finding the minimum difference
        between any two consecutive elements in the sorted list.
        """
        if len(nums) < 2:
            # By constraints this won't happen, but guard anyway.
            return 0

        nums.sort()
        # Initialize result with a large number
        res = float('inf')
        # Scan adjacent differences
        for i in range(len(nums) - 1):
            diff = nums[i+1] - nums[i]
            if diff < res:
                res = diff
        return res