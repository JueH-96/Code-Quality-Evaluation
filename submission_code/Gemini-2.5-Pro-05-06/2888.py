import collections
from typing import List

class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        n = len(nums)

        # Step 1: Find the dominant element of nums and its total frequency.
        # The problem guarantees that nums has exactly one dominant element.
        counts = collections.Counter(nums)
        
        dom_val = -1  # Placeholder, will be updated. nums[i] >= 1.
        total_freq_dom_val = 0
        
        # Identify the dominant element.
        for val, freq in counts.items():
            if freq * 2 > n:
                dom_val = val
                total_freq_dom_val = freq
                break
        
        # If dom_val remained -1, it would mean no dominant element was found,
        # contradicting the problem statement. So, dom_val is guaranteed to be set.

        # Step 2: Iterate through possible split indices i.
        # i ranges from 0 to n-2 to ensure both subarrays are non-empty.
        current_freq_dom_val_left = 0
        for i in range(n - 1):  # i from 0 to n-2
            # nums[i] is the last element of the current left subarray nums[0...i]
            if nums[i] == dom_val:
                current_freq_dom_val_left += 1
            
            len_left = i + 1
            # Check if dom_val is dominant in the left subarray
            is_dominant_in_left = (current_freq_dom_val_left * 2 > len_left)
            
            # If not dominant in left, this split is invalid.
            if not is_dominant_in_left:
                continue 
            
            # If dominant in left, check the right subarray.
            # Frequency of dom_val in right subarray:
            freq_dom_val_right = total_freq_dom_val - current_freq_dom_val_left
            len_right = n - (i + 1) # length of nums[i+1...n-1]
            
            # Check if dom_val is dominant in the right subarray
            is_dominant_in_right = (freq_dom_val_right * 2 > len_right)
            
            if is_dominant_in_right:
                # Both conditions met. This is the first such i, so it's minimal.
                return i
                
        # If loop completes, no valid split index was found.
        return -1