from typing import List
from collections import Counter

class Solution:
    def isGood(self, nums: List[int]) -> bool:
        n = max(nums)
        if len(nums) != n + 1:
            return False
        counts = Counter(nums)
        for i in range(1, n):
            if counts[i] != 1:
                return False
        if counts[n] != 2:
            return False
        return True