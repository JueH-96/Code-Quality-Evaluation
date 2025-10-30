from typing import List
from collections import Counter

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        freq = Counter(nums)
        
        def operations(f):
            max_b = f // 3
            if max_b >= 0:
                if max_b % 2 == f % 2:
                    b = max_b
                else:
                    b = max_b - 1
                    if b < 0:
                        return -1
                    if b % 2 != f % 2:
                        return -1
                a = (f - 3 * b) // 2
                if 2 * a + 3 * b != f:
                    return -1
                return a + b
            else:
                return -1
        
        total_ops = 0
        for f in freq.values():
            ops = operations(f)
            if ops == -1:
                return -1
            total_ops += ops
        return total_ops