from typing import List
from collections import Counter

class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        n = len(nums)
        # Count frequencies and find the global dominant element
        freq = Counter(nums)
        for num, count in freq.items():
            if count * 2 > n:
                dominant = num
                total_dom = count
                break
        
        prefix_dom = 0
        # Try every split point i in [0..n-2]
        for i, x in enumerate(nums):
            if x == dominant:
                prefix_dom += 1
            # Check only valid split points
            if i < n - 1:
                left_len = i + 1
                right_len = n - 1 - i
                left_dom = prefix_dom
                right_dom = total_dom - prefix_dom
                if left_dom * 2 > left_len and right_dom * 2 > right_len:
                    return i
        return -1