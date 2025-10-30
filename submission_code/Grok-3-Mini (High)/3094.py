from typing import List
from collections import Counter

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        count = Counter(nums)
        ops = 0
        for freq in count.values():
            if freq == 1:
                return -1
            ops += (freq + 2) // 3
        return ops