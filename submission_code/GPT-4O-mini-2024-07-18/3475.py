class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        operations = 0
        
        for i in range(n - 2):
            if nums[i] == 0:
                # Flip the current and the next two elements
                nums[i] ^= 1
                nums[i + 1] ^= 1
                nums[i + 2] ^= 1
                operations += 1
        
        # Check if all elements are 1
        if all(num == 1 for num in nums):
            return operations
        else:
            return -1