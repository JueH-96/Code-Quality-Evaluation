from typing import List

class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        operations = 0
        prefix_diff = 0
        
        for n, t in zip(nums, target):
            prefix_diff += t - n
            operations += abs(prefix_diff)
        
        return operations