from typing import List

class Solution:
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        """
        Performs k operations on the list nums.
        In each operation, finds the minimum value (first occurrence if duplicates)
        and replaces it with the value multiplied by multiplier.

        Args:
            nums: The input integer array.
            k: The number of operations to perform.
            multiplier: The value to multiply the minimum by.

        Returns:
            The integer array after performing all k operations.
        """
        # Perform k operations
        for _ in range(k):
            # Find the minimum value and the index of its first occurrence
            # Initialize with the first element
            min_val = nums[0]
            min_idx = 0

            # Iterate through the rest of the list to find the minimum
            # and its first index.
            for i in range(1, len(nums)):
                if nums[i] < min_val:
                    min_val = nums[i]
                    min_idx = i
                # Note: If nums[i] == min_val, we do NOT update min_idx.
                # This ensures we keep the index of the FIRST occurrence.

            # Replace the selected minimum value with x * multiplier
            nums[min_idx] = min_val * multiplier

        # Return the final state of the array
        return nums