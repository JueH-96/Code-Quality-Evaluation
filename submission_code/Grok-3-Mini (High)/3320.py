from typing import List

class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        sum_target = nums[0] + nums[1]
        operations = 1
        index = 2
        while index + 1 < len(nums) and nums[index] + nums[index + 1] == sum_target:
            operations += 1
            index += 2
        return operations