class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        operations = 0
        while True:
            # Check if all elements are distinct
            if len(set(nums)) == len(nums):
                return operations
            # Perform the operation: remove the first 3 elements
            if len(nums) >= 3:
                nums = nums[3:]
            else:
                nums = []
            operations += 1