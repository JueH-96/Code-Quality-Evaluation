from collections import defaultdict
from typing import List

class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        freq = defaultdict(int)
        left = 0
        max_length = 0
        
        for right in range(len(nums)):
            freq[nums[right]] += 1
            
            # While the current window is not valid, shrink from the left.
            while freq[nums[right]] > k:
                freq[nums[left]] -= 1
                left += 1
            
            # Update the max_length as current window is valid.
            max_length = max(max_length, right - left + 1)
        
        return max_length