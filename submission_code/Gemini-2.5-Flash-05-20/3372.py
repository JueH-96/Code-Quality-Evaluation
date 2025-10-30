from typing import List

class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        n = len(nums)

        # Base case: if the array has 0 or 1 element,
        # the longest monotonic subarray is the array itself.
        if n <= 1:
            return n

        # Initialize max_len to 1, as a single element is always a monotonic subarray.
        max_len = 1
        
        # current_increasing_len tracks the length of the strictly increasing subarray
        # ending at the current position.
        current_increasing_len = 1
        
        # current_decreasing_len tracks the length of the strictly decreasing subarray
        # ending at the current position.
        current_decreasing_len = 1

        # Iterate from the second element (index 1) to the end of the array.
        for i in range(1, n):
            # Check for strictly increasing sequence
            if nums[i] > nums[i-1]:
                current_increasing_len += 1
            else:
                # If the current element is not greater than the previous one,
                # the strictly increasing sequence is broken. Reset its length to 1.
                current_increasing_len = 1

            # Check for strictly decreasing sequence
            if nums[i] < nums[i-1]:
                current_decreasing_len += 1
            else:
                # If the current element is not less than the previous one,
                # the strictly decreasing sequence is broken. Reset its length to 1.
                current_decreasing_len = 1
            
            # Update the overall maximum length found so far
            # by comparing with both current increasing and decreasing lengths.
            max_len = max(max_len, current_increasing_len, current_decreasing_len)
        
        return max_len