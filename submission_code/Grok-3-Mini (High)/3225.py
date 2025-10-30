from typing import List
import collections

class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        freq = collections.defaultdict(int)
        left = 0
        max_length = 0
        for right in range(len(nums)):
            # Add the current element to the frequency map
            freq[nums[right]] += 1
            
            # Shrink the window from the left until no element has frequency greater than k
            while freq[nums[right]] > k:
                freq[nums[left]] -= 1
                left += 1
            
            # The window is now valid, update the maximum length
            current_len = right - left + 1
            max_length = max(max_length, current_len)
        
        return max_length