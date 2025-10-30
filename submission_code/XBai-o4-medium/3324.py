from typing import List
from collections import Counter

class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        counts = Counter(nums)
        for v in counts.values():
            if v > 2:
                return False
        return True