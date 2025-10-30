import collections
from typing import List

class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        
        # Helper function to calculate maximum subarray sum using Kadane's algorithm.
        # This version correctly handles arrays with all negative numbers,
        # ensuring a non-empty subarray is chosen.
        def calculate_kadane(arr: List[int]) -> int:
            # If the array is empty, no valid non-empty subarray can be formed.
            # Return a very small number to ensure it won't be chosen as a maximum
            # unless all other options are also invalid or similarly small.
            if not arr:
                return -float('inf') 
            
            # Initialize max_so_far and current_max with the first element.
            # This handles cases where all numbers are negative, ensuring the
            # largest negative number is returned as the max subarray sum.
            max_so_far = arr[0]
            current_max = arr[0]
            
            # Iterate from the second element
            for i in range(1, len(arr)):
                # For the current element, either start a new subarray or extend the current one.
                current_max = max(arr[i], current_max + arr[i])
                # Update the overall maximum found so far.
                max_so_far = max(max_so_far, current_max)
                
            return max_so_far

        # Case 1: No operation performed.
        # Calculate the maximum subarray sum for the original array.
        max_overall_sum = calculate_kadane(nums)

        # Get all unique values in the array. These are the candidates for removal.
        unique_values = set(nums)

        # Case 2: Perform one operation (remove all occurrences of a unique value).
        for x in unique_values:
            # Create a new array by filtering out all occurrences of x.
            current_nums = []
            for num in nums:
                if num != x:
                    current_nums.append(num)
            
            # As per the problem constraint, the array must remain non-empty after removal.
            if not current_nums:
                continue # If removing x makes the array empty, this operation is not allowed.
            
            # Calculate the maximum subarray sum for the modified array.
            current_max_sum = calculate_kadane(current_nums)
            
            # Update the overall maximum sum found so far.
            max_overall_sum = max(max_overall_sum, current_max_sum)
            
        return max_overall_sum