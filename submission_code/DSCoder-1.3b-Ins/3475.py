class Solution:
    def minOperations(self, nums: List[int]) -> int:
        # Initialize the count of operations
        operations = 0
        
        # Iterate over the array
        for i in range(1, len(nums)):
            # If the current number is not equal to the previous one
            if nums[i] != nums[i-1]:
                # Increment the count of operations
                operations += 1
        
        # Return the count of operations
        return operations