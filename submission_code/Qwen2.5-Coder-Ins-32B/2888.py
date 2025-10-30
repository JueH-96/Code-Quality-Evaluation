from collections import Counter
from typing import List

class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        total_count = Counter(nums)
        dominant_element = max(total_count, key=total_count.get)
        dominant_count = total_count[dominant_element]
        n = len(nums)
        
        left_count = 0
        for i, num in enumerate(nums):
            if num == dominant_element:
                left_count += 1
                right_count = dominant_count - left_count
                if left_count * 2 > i + 1 and right_count * 2 > n - i - 1:
                    return i
        return -1