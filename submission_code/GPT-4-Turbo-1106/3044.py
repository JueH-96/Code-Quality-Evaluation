class Solution:
    def minOperations(self, nums: list[int], k: int) -> int:
        # Create a set to store the unique elements we need to collect
        needed_elements = set(range(1, k + 1))
        
        # Initialize the number of operations to 0
        operations = 0
        
        # Iterate over the array in reverse order
        for num in reversed(nums):
            # Increment the number of operations
            operations += 1
            
            # If the current number is in the needed elements, remove it
            if num in needed_elements:
                needed_elements.remove(num)
            
            # If we have collected all needed elements, break the loop
            if not needed_elements:
                break
        
        # Return the number of operations needed
        return operations