import math  # Not strictly necessary as float('inf') is built-in
from typing import List # Necessary for type hints

class Solution:
    """
    Implements the solution to find the minimum number of operations
    to make the array non-decreasing following the specified procedure.

    The procedure involves repeatedly finding the adjacent pair with the minimum sum
    (leftmost in case of ties) and replacing that pair with their sum.
    """
    def minimumPairRemoval(self, nums: List[int]) -> int:
        """
        Calculates the minimum number of operations to make the array non-decreasing
        by repeatedly merging the adjacent pair with the minimum sum (leftmost tie-break).

        Args:
            nums: The input list of integers. Constraints: 1 <= len(nums) <= 50, -1000 <= nums[i] <= 1000.

        Returns:
            The minimum number of operations required according to the specified process.
        """
        
        # Make a mutable copy of the list to avoid modifying the original input array.
        current_nums = list(nums)
        count = 0  # Initialize the operation counter
        
        while True:
            n = len(current_nums)
            
            # Check if the current array is non-decreasing.
            # An array is non-decreasing if each element is greater than or equal 
            # to its previous element (if one exists).
            # Arrays with 0 or 1 element are considered non-decreasing.
            is_non_decreasing = True
            if n > 1: 
                for i in range(n - 1):
                    if current_nums[i] > current_nums[i+1]:
                        is_non_decreasing = False
                        break
            
            # If the array is non-decreasing, the process stops. Return the total count of operations.
            if is_non_decreasing:
                return count

            # If the array is not non-decreasing, we must perform one operation.
            # Since the non-decreasing check passed for n <= 1, we know here that n >= 2.

            # Find the adjacent pair (current_nums[i], current_nums[i+1]) with the minimum sum.
            # If multiple pairs have the same minimum sum, the problem states to choose the leftmost one.
            min_sum = float('inf')  # Initialize minimum sum to positive infinity
            min_index = -1         # Initialize index of the start of the minimum pair

            for i in range(n - 1):
                current_sum = current_nums[i] + current_nums[i+1]
                
                # If we find a sum strictly smaller than the current minimum, update min_sum and min_index.
                # If we find a sum equal to the current minimum, we do *not* update min_index.
                # This ensures that we always keep the index of the *first* (leftmost) occurrence 
                # of the minimum sum found so far.
                if current_sum < min_sum:
                    min_sum = current_sum
                    min_index = i
            
            # Perform the replacement operation:
            # Calculate the sum of the chosen pair.
            new_sum = current_nums[min_index] + current_nums[min_index+1]
            
            # Update the list `current_nums` in place.
            # It's important to remove the element at the higher index first to avoid index shifting issues.
            current_nums.pop(min_index + 1) 
            # Now, update the element at the lower index (min_index) with the calculated sum.
            current_nums[min_index] = new_sum
            
            # Increment the operation counter as one operation has been performed.
            count += 1
            
            # The loop continues with the modified array.
            # Since each operation reduces the length of the array by 1,
            # the process is guaranteed to terminate eventually (at the latest when the array has length 1).