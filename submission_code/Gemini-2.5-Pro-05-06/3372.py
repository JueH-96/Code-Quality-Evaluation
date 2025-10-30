from typing import List

class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Constraints: 1 <= nums.length <= 50. So n is always at least 1.
        
        # If the array has only one element, that element itself forms a
        # monotonic subarray of length 1. This is the longest possible.
        if n == 1:
            return 1

        # Initialize max_overall_length to 1.
        # This is because any single element is a monotonic subarray of length 1.
        # This also serves as the correct default if no longer sequences are found
        # (e.g., for an array like [3, 3, 3]).
        max_overall_length = 1

        # Pass 1: Find the length of the longest strictly increasing subarray
        current_length = 1
        for i in range(1, n):
            if nums[i] > nums[i-1]:
                current_length += 1
            else:
                # The strictly increasing sequence broke (nums[i] <= nums[i-1]).
                # A new potential increasing subarray starts with nums[i], so its length is 1.
                current_length = 1
            max_overall_length = max(max_overall_length, current_length)
        
        # Pass 2: Find the length of the longest strictly decreasing subarray
        # Reset current_length for the decreasing pass.
        current_length = 1
        for i in range(1, n):
            if nums[i] < nums[i-1]:
                current_length += 1
            else:
                # The strictly decreasing sequence broke (nums[i] >= nums[i-1]).
                # A new potential decreasing subarray starts with nums[i], so its length is 1.
                current_length = 1
            # Update max_overall_length with the maximum length found so far.
            # This ensures it captures the longest from either increasing or decreasing sequences.
            max_overall_length = max(max_overall_length, current_length)
            
        return max_overall_length