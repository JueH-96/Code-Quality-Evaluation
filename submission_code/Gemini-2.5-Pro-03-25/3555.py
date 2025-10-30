import math
from typing import List

class Solution:
    """
    Solves the problem of repeatedly finding the minimum element in a list, 
    multiplying its first occurrence by a multiplier, and repeating this k times.
    """
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        """
        Performs k operations on nums. In each operation:
        1. Find the minimum value x in nums. Select the first occurrence if duplicates exist.
        2. Replace the selected minimum value x with x * multiplier.
        Returns the final state of nums.

        Args:
            nums: The initial list of integers.
            k: The number of operations to perform.
            multiplier: The factor to multiply the minimum element by in each operation.

        Returns:
            The list nums after performing k operations.
        """

        # Check if the multiplier is 1. If so, the list state won't change 
        # after the first minimum is found (unless k=0, but k>=1).
        # However, the loop below handles this case correctly without special checks.
        # If k=0, the loop range(0) is empty, and the original nums is returned, which is correct.
        
        if not nums: # Handle empty list case, though constraints say len >= 1
             return []

        for _ in range(k):
            # Find the minimum value in the current list
            min_val = min(nums)
            
            # Find the index of the *first* occurrence of the minimum value.
            # The list.index() method naturally finds the first occurrence.
            try:
                min_index = nums.index(min_val)
            except ValueError:
                # This should theoretically not happen if nums is not empty,
                # as min() would have found a value present in the list.
                # But it's good practice to handle potential exceptions.
                # If somehow the minimum value isn't found (e.g., due to concurrency issues
                # in a different context, though not applicable here), we might stop.
                break 

            # Replace the element at the found index with its value multiplied by the multiplier
            nums[min_index] *= multiplier
            
        # Return the modified list after k operations
        return nums