from collections import Counter
from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        count = Counter(nums)
        total = 0
        for c in count.values():
            if c % 3 == 0:
                total += c // 3
            elif c % 3 == 1:
                if c // 3 >= 1:
                    total += (c // 3 - 1) + 2
                else:
                    return -1
            elif c % 3 == 2:
                total += (c // 3) + 1
            else:
                return -1
        return total