class Solution:
    def minOperations(self, k: int) -> int:
        if k == 1:
            return 0
        min_ops = float('inf')
        for m in range(1, k + 1):
            x = (k + m - 1) // m  # Equivalent to ceil(k / m)
            ops = m + x - 2
            if ops < min_ops:
                min_ops = ops
        return min_ops