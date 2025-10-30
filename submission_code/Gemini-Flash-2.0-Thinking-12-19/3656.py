from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        operations = 0
        while True:
            if len(set(nums)) == len(nums):
                return operations
            if len(nums) >= 3:
                nums = nums[3:]
            else:
                nums = []
            operations += 1