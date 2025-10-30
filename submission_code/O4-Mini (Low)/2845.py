from typing import List

class Solution:
    def findValueOfPartition(self, nums: List[int]) -> int:
        # Sort the array so that the minimum gap between adjacent elements
        # gives the minimum possible |max(nums1) - min(nums2)|.
        nums.sort()
        # Initialize the minimum difference with a large value.
        min_diff = float('inf')
        # Scan adjacent pairs to find the smallest difference.
        for i in range(1, len(nums)):
            diff = nums[i] - nums[i - 1]
            if diff < min_diff:
                min_diff = diff
        return min_diff