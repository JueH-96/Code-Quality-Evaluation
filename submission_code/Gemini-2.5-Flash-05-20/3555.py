from typing import List

class Solution:
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        """
        Performs k operations on the given list of numbers.
        In each operation, it finds the minimum value, selects its first occurrence,
        and replaces it with the value multiplied by the given multiplier.

        Args:
            nums: A list of integers.
            k: The number of operations to perform.
            multiplier: The value to multiply the minimum element by.

        Returns:
            The list of integers after k operations.
        """
        
        # Perform k operations as specified.
        for _ in range(k):
            min_val = float('inf')  # Initialize min_val to positive infinity.
                                    # This ensures that the first element (or any element)
                                    # of nums will be smaller than min_val, correctly setting it.
            min_idx = -1            # Initialize min_idx to an invalid index.
                                    # This will be updated to the index of the found minimum.

            # Iterate through the array to find the minimum value and its first occurrence index.
            # We must iterate from left to right to ensure we find the *first* occurrence
            # if multiple elements have the same minimum value.
            for i in range(len(nums)):
                if nums[i] < min_val:
                    # If a new minimum value is found, update min_val and its index.
                    min_val = nums[i]
                    min_idx = i
                # If nums[i] == min_val, we do not update min_idx.
                # This ensures that min_idx always points to the *first* encountered
                # occurrence of the current minimum value.

            # After the loop, min_idx will hold the index of the first occurrence
            # of the minimum value in the current state of `nums`.
            # The problem constraints guarantee `nums` is non-empty, so `min_idx` will always be valid.
            
            # Replace the selected minimum value with x * multiplier.
            nums[min_idx] *= multiplier
            
        # After completing all k operations, return the modified array.
        return nums