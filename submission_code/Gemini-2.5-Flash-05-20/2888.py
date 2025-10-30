import collections
from typing import List

class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        n = len(nums)

        # Step 1: Find the dominant element of the entire array and its total frequency.
        # The problem guarantees that `nums` has exactly one dominant element.
        counts = collections.Counter(nums)
        
        main_dominant_val = -1
        total_freq = 0
        
        # Iterate through the counts to find the unique dominant element.
        # Its frequency `freq` must satisfy `freq * 2 > n`.
        for num, freq in counts.items():
            if freq * 2 > n:
                main_dominant_val = num
                total_freq = freq
                break # Found it, no need to check other elements

        # This check is mostly for robustness, though problem constraints imply 
        # main_dominant_val will always be found.
        if main_dominant_val == -1:
            return -1 

        # Step 2: Iterate through all possible split points `i`.
        # A valid split index `i` must satisfy `0 <= i < n - 1`.
        # The left subarray is nums[0, ..., i] with length (i + 1).
        # The right subarray is nums[i + 1, ..., n - 1] with length (n - (i + 1)).

        current_left_freq = 0 # Frequency of main_dominant_val in nums[0 ... i]

        # Iterate `i` from 0 up to n - 2 (inclusive)
        for i in range(n - 1):
            # Update the frequency of the main_dominant_val in the current left subarray
            if nums[i] == main_dominant_val:
                current_left_freq += 1
            
            len_left = i + 1 # Length of the left subarray
            
            # Frequency of main_dominant_val in the right subarray
            current_right_freq = total_freq - current_left_freq
            
            len_right = n - len_left # Length of the right subarray

            # Check if the main_dominant_val is dominant in both subarrays
            is_left_dominant = (current_left_freq * 2 > len_left)
            is_right_dominant = (current_right_freq * 2 > len_right)
            
            # If both conditions are met, `i` is a valid split index.
            # Since we iterate `i` from left to right, the first one found will be the minimum.
            if is_left_dominant and is_right_dominant:
                return i
                
        # If the loop completes, no valid split index was found.
        return -1