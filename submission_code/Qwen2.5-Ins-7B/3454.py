class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        operations = 0
        nums.sort()
        target.sort()
        
        for i in range(len(nums)):
            if nums[i] != target[i]:
                operations += abs(nums[i] - target[i])
                nums[i] = target[i]
        
        return operations