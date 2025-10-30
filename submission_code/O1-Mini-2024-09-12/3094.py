from typing import List
from collections import Counter

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        counts = Counter(nums)
        total_ops = 0
        for c in counts.values():
            k_max = c // 3
            found = False
            for k in range(k_max, k_max - 3, -1):
                if k < 0:
                    continue
                remainder = c - 3 * k
                if remainder % 2 == 0:
                    total_ops += k + (remainder // 2)
                    found = True
                    break
            if not found:
                return -1
        return total_ops