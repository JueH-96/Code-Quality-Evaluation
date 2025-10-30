class Solution:
    def minOperations(self, k: int) -> int:
        import math
        min_ops = float('inf')
        x0 = int(math.sqrt(k - 1)) if k > 1 else 1
        for x in range(max(1, x0 - 3), x0 + 4):
            ops = (x - 1) + (k - 1) // x
            if ops < min_ops:
                min_ops = ops
        return min_ops