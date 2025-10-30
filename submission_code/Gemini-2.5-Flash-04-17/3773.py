from typing import List

class Solution:
    def is_non_decreasing(self, arr: List[int]) -> bool:
        """Checks if the array is non-decreasing."""
        # An array with 0 or 1 element is considered non-decreasing
        if len(arr) <= 1:
            return True
        for i in range(len(arr) - 1):
            if arr[i] > arr[i+1]:
                return False
        return True

    def minimumPairRemoval(self, nums: List[int]) -> int:
        """
        Given an array nums, perform operations to make it non-decreasing.
        Operation: Select the leftmost adjacent pair with the minimum sum, replace with sum.
        Return the minimum number of operations needed.
        """
        operations = 0
        current_nums = list(nums) # Work on a copy

        # The loop continues as long as the array is not non-decreasing.
        # An array with 0 or 1 element is non-decreasing, so the loop condition naturally handles base cases.
        while not self.is_non_decreasing(current_nums):
            # Find the minimum adjacent sum and its leftmost index
            # We are guaranteed len(current_nums) >= 2 when we enter the loop body
            # because arrays of length 0 or 1 are always non-decreasing.
            
            min_sum = current_nums[0] + current_nums[1]
            min_index = 0

            # Iterate from the second pair onwards (if any)
            # The range is up to len(current_nums) - 1, which means checking indices i from 0 up to len-2.
            # We already handled i=0 initially.
            for i in range(1, len(current_nums) - 1):
                current_sum = current_nums[i] + current_nums[i+1]
                
                # If a strictly smaller sum is found, update min_sum and min_index.
                # If an equal sum is found, we do nothing, thereby ensuring min_index
                # corresponds to the leftmost occurrence of the minimum sum found so far.
                if current_sum < min_sum:
                    min_sum = current_sum
                    min_index = i

            # Perform the merge operation: replace the pair at min_index with their sum (min_sum)
            
            # Construct the new list using slicing and concatenation.
            # This is clear and reasonably efficient for the given constraints (N=50).
            # It creates a new list object in each iteration.
            current_nums = current_nums[:min_index] + [min_sum] + current_nums[min_index + 2:]

            # Increment the count of operations performed
            operations += 1

        return operations