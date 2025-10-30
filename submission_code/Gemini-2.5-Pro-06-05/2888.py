import collections
from typing import List

class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        
        n = len(nums)

        # Step 1: Find the dominant element and its total frequency.
        # The problem guarantees that `nums` has exactly one dominant element.
        # An element `x` is dominant if `freq(x) * 2 > m`, where `m` is the array length.
        
        # We can find the dominant element by counting all frequencies.
        counts = collections.Counter(nums)
        
        dominant_elem = -1
        total_freq = 0
        
        # There is only one element `x` for which `freq(x) * 2 > n`.
        for num, freq in counts.items():
            if freq * 2 > n:
                dominant_elem = num
                total_freq = freq
                break
        
        # Step 2: Iterate through all possible split points and check for validity.
        # A split at index 'i' (where 0 <= i < n - 1) is valid if the dominant element
        # of the whole array is also dominant in the left subarray `nums[0...i]` and
        # the right subarray `nums[i+1...n-1]`.
        
        left_count = 0
        for i in range(n - 1): # i can be from 0 to n-2
            # Update the count of the dominant element in the left subarray
            if nums[i] == dominant_elem:
                left_count += 1
            
            # Check the dominance condition for the left subarray
            # Left subarray is `nums[0...i]`, its length is `i + 1`
            left_len = i + 1
            is_dominant_in_left = (left_count * 2) > left_len
            
            # Check the dominance condition for the right subarray
            # Right subarray is `nums[i+1...n-1]`, its length is `n - (i + 1)`
            right_len = n - left_len
            right_count = total_freq - left_count
            is_dominant_in_right = (right_count * 2) > right_len
            
            # If both subarrays have the same dominant element (which must be `dominant_elem`),
            # we've found the minimum valid split index because we iterate from left to right.
            if is_dominant_in_left and is_dominant_in_right:
                return i
                
        # Step 3: If the loop completes, it means no valid split exists.
        return -1