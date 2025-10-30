class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        operations = 0
        
        # Process from left to right
        for i in range(n - 2):
            # If current element is 0, we need to flip
            if nums[i] == 0:
                # Flip current and next 2 elements
                nums[i] = 1 - nums[i]
                nums[i + 1] = 1 - nums[i + 1]
                nums[i + 2] = 1 - nums[i + 2]
                operations += 1
        
        # Check if all elements are 1
        for i in range(n):
            if nums[i] == 0:
                return -1
        
        return operations