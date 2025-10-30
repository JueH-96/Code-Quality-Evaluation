import math

class Solution:
    def minOperations(self, k: int) -> int:
        if k == 1:
            return 0
        min_ops = float('inf')
        max_x = int(math.isqrt(k)) + 1
        for x in range(1, max_x + 1):
            m = (k + x - 1) // x - 1
            ops = (x - 1) + m
            if ops < min_ops:
                min_ops = ops
        return min_ops