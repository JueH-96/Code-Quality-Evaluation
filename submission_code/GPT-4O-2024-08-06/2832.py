from collections import defaultdict
from typing import List

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        # Dictionary to store the frequency of each number
        freq = defaultdict(int)
        max_freq = 0
        left = 0
        
        # Sliding window approach
        for right in range(len(nums)):
            # Increment the frequency of the current number
            freq[nums[right]] += 1
            # Update the maximum frequency found so far
            max_freq = max(max_freq, freq[nums[right]])
            
            # Calculate the number of deletions needed to make the current window equal
            # If the deletions exceed k, shrink the window from the left
            if (right - left + 1) - max_freq > k:
                freq[nums[left]] -= 1
                left += 1
        
        # The length of the longest equal subarray is the size of the window
        return len(nums) - left