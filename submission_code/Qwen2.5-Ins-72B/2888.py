from collections import Counter
from typing import List

class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        # Find the dominant element in the entire array
        freq = Counter(nums)
        n = len(nums)
        dominant = max(freq, key=freq.get)
        
        # Check if the dominant element is valid for the entire array
        if freq[dominant] * 2 <= n:
            return -1
        
        # Track the count of the dominant element in the left and right subarrays
        left_count = 0
        right_count = freq[dominant]
        
        for i in range(n - 1):
            if nums[i] == dominant:
                left_count += 1
                right_count -= 1
            
            # Check if the dominant element is valid in both subarrays
            if left_count * 2 > (i + 1) and right_count * 2 > (n - i - 1):
                return i
        
        return -1