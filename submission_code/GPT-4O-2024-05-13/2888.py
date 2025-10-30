from typing import List
from collections import Counter

class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        n = len(nums)
        freq = Counter(nums)
        
        # Find the dominant element
        dominant = None
        for key, value in freq.items():
            if value * 2 > n:
                dominant = key
                break
        
        if dominant is None:
            return -1
        
        left_count = 0
        right_count = freq[dominant]
        
        for i in range(n - 1):
            if nums[i] == dominant:
                left_count += 1
                right_count -= 1
            
            if left_count * 2 > (i + 1) and right_count * 2 > (n - i - 1):
                return i
        
        return -1