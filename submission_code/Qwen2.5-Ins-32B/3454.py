from typing import List

class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        operations = 0
        diff = 0
        for i in range(len(nums)):
            diff = target[i] - nums[i]
            if diff != 0:
                operations += abs(diff)
                nums[i] = target[i]
        return operations