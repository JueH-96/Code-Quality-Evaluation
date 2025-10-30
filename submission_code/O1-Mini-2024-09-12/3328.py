class Solution:
    def minOperations(self, k: int) -> int:
        if k == 1:
            return 0
        min_ops = float('inf')
        for x in range(1, k + 1):
            d = (k + x - 1) // x - 1
            ops = (x - 1) + d
            if ops < min_ops:
                min_ops = ops
            if x > k // 2 + 1:
                break
        return min_ops