from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        seen = set()
        operations = 0
        i = 0
        
        while i < len(nums):
            # If the current number is already seen, we need to perform an operation
            if nums[i] in seen:
                operations += 1
                # Remove the next 3 elements or all remaining if less than 3
                i += 3
            else:
                # Add the current number to the seen set
                seen.add(nums[i])
                i += 1
        
        # If there are still elements left that are not distinct, we may need more operations
        while i < len(nums):
            operations += 1
            i += 3
        
        return operations