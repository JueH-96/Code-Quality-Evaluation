from collections import Counter
from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        counts = Counter(nums)
        res = 0
        for v in counts.values():
            if v == 1:
                return -1
            res += (v + 2) // 3
        return res