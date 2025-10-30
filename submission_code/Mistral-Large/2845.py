from typing import List

class Solution:
    def findValueOfPartition(self, nums: List[int]) -> int:
        # Sort the array
        nums.sort()

        # Initialize the minimum difference to a large value
        min_diff = float('inf')

        # Iterate through the sorted array and find the minimum difference between consecutive elements
        for i in range(1, len(nums)):
            min_diff = min(min_diff, nums[i] - nums[i - 1])

        return min_diff