from typing import List
from collections import Counter

class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        freq = Counter(nums)
        max_freq = max(freq.values(), default=0)
        if max_freq > 2:
            return False
        count_twice = sum(1 for v in freq.values() if v == 2)
        return count_twice <= len(nums) // 2