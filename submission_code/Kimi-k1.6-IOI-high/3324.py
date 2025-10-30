from collections import Counter
from typing import List

class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        count = Counter(nums)
        max_freq = max(count.values())
        if max_freq > 2:
            return False
        unique_count = len(count)
        required = len(nums) // 2
        return unique_count >= required