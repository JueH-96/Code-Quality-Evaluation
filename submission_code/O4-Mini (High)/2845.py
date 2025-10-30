from typing import List

class Solution:
    def findValueOfPartition(self, nums: List[int]) -> int:
        # Sort the array to bring closest elements together.
        nums.sort()
        # Initialize the minimum difference with a large value.
        min_diff = float('inf')
        # Iterate through adjacent pairs to find the smallest gap.
        for i in range(len(nums) - 1):
            diff = nums[i+1] - nums[i]
            if diff < min_diff:
                min_diff = diff
        return min_diff