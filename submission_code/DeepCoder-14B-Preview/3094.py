from collections import Counter
from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        freq = Counter(nums)
        total_ops = 0
        for count in freq.values():
            max_b = count // 3
            min_ops = None
            for b in range(max_b, -1, -1):
                remainder = count - 3 * b
                if remainder < 0:
                    continue
                if remainder % 2 == 0:
                    a = remainder // 2
                    min_ops = a + b
                    break
            if min_ops is None:
                return -1
            total_ops += min_ops
        return total_ops