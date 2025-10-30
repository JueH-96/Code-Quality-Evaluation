from typing import List
from collections import Counter

class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        n = len(nums)
        freq = Counter(nums)
        dom = -1
        total_dom = 0
        
        # Find the dominant element
        for num, count in freq.items():
            if count * 2 > n:
                dom = num
                total_dom = count
                break
        
        # Compute prefix sums for the dominant element
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + (1 if nums[i] == dom else 0)
        
        # Check each possible split index
        for i in range(n - 1):
            left_count = prefix[i + 1]
            left_len = i + 1
            if left_count * 2 > left_len:
                right_count = total_dom - left_count
                right_len = n - (i + 1)
                if right_count * 2 > right_len:
                    return i
        
        return -1