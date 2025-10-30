from typing import List
from collections import Counter

class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        counts = Counter(nums)
        n = len(nums) // 2
        if any(v > 2 for v in counts.values()):
            return False
        return len(counts) >= n