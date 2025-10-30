import collections
from typing import List

class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        
        def kadane(arr: List[int]) -> int:
            """
            Calculates the maximum subarray sum for a non-empty array using Kadane's algorithm.
            """
            # This function will be called on non-empty arrays based on the main logic.
            max_so_far = -float('inf')
            current_max = -float('inf')
            
            for x in arr:
                current_max = max(x, current_max + x)
                max_so_far = max(max_so_far, current_max)
            return max_so_far

        # Case 1: No operation is performed.
        # The result must be at least the max subarray sum of the original array.
        max_sum = kadane(nums)

        # Get the unique values. We will consider removing each one.
        unique_vals = set(nums)
        
        # If there's only one unique number in the array, removing it would
        # make the array empty, which is not allowed by the problem statement.
        # In this scenario, no operation can be performed.
        if len(unique_vals) == 1:
            return max_sum
        
        # Case 2: One operation is performed.
        # Iterate through each unique value, remove it, and find the max subarray sum.
        for x in unique_vals:
            # Create a new array by filtering out all occurrences of x.
            # This is a direct implementation of the "remove" operation.
            temp_nums = [num for num in nums if num != x]
            
            # Since we've established len(unique_vals) > 1, temp_nums is
            # guaranteed to be non-empty.
            max_sum = max(max_sum, kadane(temp_nums))
            
        return max_sum