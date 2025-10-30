import collections
from typing import List

class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        freq_map = collections.Counter()  # To store frequencies of elements in the current window
        l = 0  # Left pointer of the sliding window
        max_length = 0  # Stores the maximum length of a good subarray found

        # Iterate with the right pointer 'r' to expand the window
        for r in range(len(nums)):
            # Add the current element nums[r] to the frequency map
            freq_map[nums[r]] += 1

            # If the frequency of the element just added (nums[r]) exceeds k,
            # the current window [l...r] is no longer "good".
            # We need to shrink the window from the left until the condition is met again.
            while freq_map[nums[r]] > k:
                # Decrease the frequency of the element at the left pointer nums[l]
                freq_map[nums[l]] -= 1
                
                # Move the left pointer one position to the right to shrink the window
                l += 1
            
            # At this point, the window nums[l...r] is guaranteed to be "good".
            # The frequency of every element within this window is less than or equal to k.
            # Calculate the current window's length and update max_length if it's larger.
            max_length = max(max_length, r - l + 1)
        
        return max_length