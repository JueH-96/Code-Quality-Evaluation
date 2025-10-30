from typing import List

class Solution:
    def findValueOfPartition(self, nums: List[int]) -> int:
        # Sort the array to bring close values next to each other.
        nums.sort()
        # Initialize the minimum difference to a large number.
        min_diff = float('inf')
        # Check adjacent differences in the sorted array.
        for i in range(len(nums) - 1):
            min_diff = min(min_diff, nums[i+1] - nums[i])
        return min_diff