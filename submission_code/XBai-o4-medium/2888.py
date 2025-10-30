from typing import List
from collections import Counter

class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        counts = Counter(nums)
        n = len(nums)
        dom_val = None
        for num, cnt in counts.items():
            if 2 * cnt > n:
                dom_val = num
                break
        
        # Compute prefix sums for the dominant value
        prefix = [0] * n
        current = 0
        for i in range(n):
            if nums[i] == dom_val:
                current += 1
            prefix[i] = current
        
        total_count = prefix[-1]
        
        for i in range(n - 1):
            left_count = prefix[i]
            left_len = i + 1
            right_count = total_count - left_count
            right_len = n - i - 1
            if 2 * left_count > left_len and 2 * right_count > right_len:
                return i
        
        return -1