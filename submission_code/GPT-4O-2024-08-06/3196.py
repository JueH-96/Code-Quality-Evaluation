from typing import List
from collections import Counter

class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        left = 0
        total = 0
        max_freq = 0
        
        for right in range(len(nums)):
            total += nums[right]
            
            # Check if we can make all elements in the window equal to nums[right]
            while nums[right] * (right - left + 1) > total + k:
                total -= nums[left]
                left += 1
            
            # Update the maximum frequency
            max_freq = max(max_freq, right - left + 1)
        
        return max_freq