from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        operations = 0
        while len(nums) > 0:
            # Check if the current list has distinct elements
            if len(nums) == len(set(nums)):
                break
            # Remove the first 3 elements
            nums = nums[3:]
            operations += 1
        return operations