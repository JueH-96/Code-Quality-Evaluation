from typing import List
from collections import Counter

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        counts = Counter(nums)
        operations = 0
        for count in counts.values():
            if count == 1:
                return -1
            elif count % 3 == 0:
                operations += count // 3
            elif count % 3 == 1:
                operations += (count - 4) // 3 + 2
            else:
                operations += (count - 2) // 3 + 1
        return operations