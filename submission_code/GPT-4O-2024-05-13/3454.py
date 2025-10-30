class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        operations = 0
        diff = 0
        
        for n, t in zip(nums, target):
            diff += t - n
            operations += abs(diff)
        
        return operations