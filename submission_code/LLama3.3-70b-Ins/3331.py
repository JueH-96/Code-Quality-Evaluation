from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Sort the array in ascending order
        nums.sort()
        
        # Initialize a variable to store the minimum number of operations
        operations = 0
        
        # Iterate over the array from the start
        for num in nums:
            # If the current number is less than k, increment the operations count
            if num < k:
                operations += 1
            # If the current number is greater than or equal to k, break the loop
            else:
                break
        
        # Return the minimum number of operations
        return operations