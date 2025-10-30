from typing import List
from collections import Counter

class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        n = len(nums)
        # Determine the overall dominant element in nums.
        total_counts = Counter(nums)
        dominant, total_freq = None, 0
        for num, freq in total_counts.items():
            if freq * 2 > n:
                dominant = num
                total_freq = freq
                break
        
        # If no overall dominant element exists (shouldn't happen per constraints), return -1.
        if dominant is None:
            return -1
        
        # Scan splits from index 0 to n-2.
        left_count = 0
        for i in range(n - 1):
            if nums[i] == dominant:
                left_count += 1
            left_length = i + 1
            right_length = n - left_length
            # Check left and right subarray dominant condition using counts.
            if left_count * 2 > left_length and (total_freq - left_count) * 2 > right_length:
                return i
        
        return -1