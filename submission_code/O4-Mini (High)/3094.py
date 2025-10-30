from typing import List
from collections import Counter

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        cnt = Counter(nums)
        ops = 0
        for v in cnt.values():
            # If any value appears exactly once, it's impossible
            if v == 1:
                return -1
            # The minimum number of operations for count v is ceil(v / 3)
            ops += (v + 2) // 3
        return ops