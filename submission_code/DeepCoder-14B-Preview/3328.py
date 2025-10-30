import math

class Solution:
    def minOperations(self, k: int) -> int:
        if k == 1:
            return 0
        x_max = int(math.isqrt(k))
        min_ops = float('inf')
        for x in range(1, x_max + 1):
            m = (k + x - 1) // x
            sum_ops = x + m - 2
            if sum_ops < min_ops:
                min_ops = sum_ops
        return min_ops