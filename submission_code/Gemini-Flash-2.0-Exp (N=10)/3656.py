class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        def is_distinct(arr):
            return len(set(arr)) == len(arr)

        if is_distinct(nums):
            return 0
        
        operations = 0
        while nums:
            operations += 1
            nums = nums[3:]
            if is_distinct(nums):
                return operations
            if not nums:
                return operations
        return operations