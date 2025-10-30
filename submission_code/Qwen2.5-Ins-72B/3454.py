from typing import List

class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        diff = [target[i] - nums[i] for i in range(len(nums))]
        operations = 0
        current_diff = 0
        
        for i in range(len(diff)):
            if diff[i] != 0:
                operations += abs(diff[i] - current_diff)
                current_diff = diff[i]
        
        return operations