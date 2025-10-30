class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        operations = 0
        while len(nums) > 0:
            if len(set(nums)) == len(nums):
                return operations
            
            if len(nums) >= 3:
                nums = nums[3:]
            else:
                nums = []
            operations += 1
        return operations