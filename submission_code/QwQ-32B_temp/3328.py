import math

class Solution:
    def minOperations(self, k: int) -> int:
        if k == 1:
            return 0
        min_ops = float('inf')
        s = int(math.sqrt(k))
        # Check x from 1 to s + 1 (inclusive)
        for x in range(1, s + 2):
            m = (k + x - 1) // x  # Equivalent to ceil(k/x)
            ops = (x - 1) + (m - 1)
            if ops < min_ops:
                min_ops = ops
        return min_ops