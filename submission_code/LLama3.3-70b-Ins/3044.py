from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Create a set to store the elements we need to collect
        needed = set(range(1, k + 1))
        
        # Initialize the count of operations
        operations = 0
        
        # Iterate over the array from the end
        for num in reversed(nums):
            # If the current number is in the set of needed elements, remove it
            if num in needed:
                needed.remove(num)
            # Increment the count of operations
            operations += 1
            # If we have collected all the needed elements, break the loop
            if not needed:
                break
        
        # Return the count of operations
        return operations