from typing import List
from collections import Counter

class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        total_count = Counter(nums)
        dominant_element = max(total_count, key=total_count.get)
        dominant_count = total_count[dominant_element]
        
        left_count = 0
        for i, num in enumerate(nums):
            if num == dominant_element:
                left_count += 1
                right_count = dominant_count - left_count
                if left_count * 2 > i + 1 and right_count * 2 > len(nums) - i - 1:
                    return i
        return -1