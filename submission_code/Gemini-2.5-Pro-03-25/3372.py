# Import List from typing for type hinting
from typing import List

class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        """
        Finds the length of the longest subarray of nums which is either
        strictly increasing or strictly decreasing.

        A subarray is a contiguous part of an array.
        A subarray is strictly increasing if each element is strictly greater than the previous one.
        A subarray is strictly decreasing if each element is strictly less than the previous one.

        Args:
            nums: A list of integers. The constraints are:
                  1 <= nums.length <= 50
                  1 <= nums[i] <= 50

        Returns:
            The length of the longest monotonic (either strictly increasing or strictly decreasing) subarray.
        """
        
        # Get the number of elements in the input list
        n = len(nums)
        
        # Handle the base case where the list has 0 or 1 element.
        # According to constraints, n >= 1. If n = 1, the longest monotonic subarray
        # is the array itself, which has length 1.
        if n <= 1:
            return n

        # Initialize the maximum length found so far. Since n > 1, there are at least
        # two elements. A single element itself forms a monotonic subarray of length 1.
        # So, the minimum possible maximum length is 1.
        max_length = 1
        
        # Initialize the length of the current strictly increasing subarray ending at the current index.
        # We start considering the array from the first element (index 0), so its length is 1 initially.
        current_increasing_length = 1
        
        # Initialize the length of the current strictly decreasing subarray ending at the current index.
        # Similarly, initialize its length to 1 for the first element.
        current_decreasing_length = 1

        # Iterate through the array starting from the second element (index 1) up to the end.
        for i in range(1, n):
            # Compare the current element `nums[i]` with the previous element `nums[i-1]`.
            
            if nums[i] > nums[i-1]:
                # If the current element is greater than the previous one:
                # - The strictly increasing sequence continues. Increment its length.
                current_increasing_length += 1
                # - The strictly decreasing sequence is broken. Reset its length to 1 
                #   (the current element `nums[i]` starts a new potential decreasing sequence).
                current_decreasing_length = 1
            elif nums[i] < nums[i-1]:
                # If the current element is less than the previous one:
                # - The strictly decreasing sequence continues. Increment its length.
                current_decreasing_length += 1
                # - The strictly increasing sequence is broken. Reset its length to 1
                #   (the current element `nums[i]` starts a new potential increasing sequence).
                current_increasing_length = 1
            else: # nums[i] == nums[i-1]
                # If the current element is equal to the previous one:
                # - Neither a strictly increasing nor a strictly decreasing sequence can continue.
                # - Reset both lengths to 1 (the current element `nums[i]` starts new potential sequences).
                current_increasing_length = 1
                current_decreasing_length = 1

            # After updating the lengths of the monotonic subarrays ending at the current index `i`,
            # update the overall maximum length found so far by comparing it with the current lengths.
            max_length = max(max_length, current_increasing_length, current_decreasing_length)

        # After iterating through all elements, return the overall maximum length found.
        return max_length