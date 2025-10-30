class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        operations = 0
        for i in range(len(nums)):
            operations += abs(nums[i] - target[i])
        return operations