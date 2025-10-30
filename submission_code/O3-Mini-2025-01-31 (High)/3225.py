from typing import List
from collections import defaultdict

class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        freq = defaultdict(int)
        left = 0
        longest = 0
        
        for right in range(len(nums)):
            freq[nums[right]] += 1
            
            # If the frequency of nums[right] becomes > k,
            # shrink the window from the left until it's valid.
            while freq[nums[right]] > k:
                freq[nums[left]] -= 1
                left += 1
            
            # Update the maximum length of a good subarray.
            longest = max(longest, right - left + 1)
            
        return longest