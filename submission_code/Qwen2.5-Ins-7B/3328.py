class Solution:
    def minOperations(self, k: int) -> int:
        if k == 1:
            return 0
        min_ops = float('inf')
        for i in range(1, k + 1):
            ops = i - 1 + (k + i - 1) // i
            min_ops = min(min_ops, ops)
        return min_ops