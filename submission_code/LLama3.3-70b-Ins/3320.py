from typing import List

class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        max_operations = 0
        while len(nums) >= 2:
            score = nums[0] + nums[1]
            nums = nums[2:]
            operations = 1
            while len(nums) >= 2 and nums[0] + nums[1] == score:
                nums = nums[2:]
                operations += 1
            max_operations = max(max_operations, operations)
        return max_operations