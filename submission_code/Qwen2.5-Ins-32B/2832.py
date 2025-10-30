from typing import List
from collections import defaultdict

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        max_len = 0
        left = 0
        count = defaultdict(int)
        
        for right in range(len(nums)):
            count[nums[right]] += 1
            max_freq = max(count.values())
            
            while (right - left + 1) - max_freq > k:
                count[nums[left]] -= 1
                left += 1
            
            max_len = max(max_len, max_freq)
        
        return max_len