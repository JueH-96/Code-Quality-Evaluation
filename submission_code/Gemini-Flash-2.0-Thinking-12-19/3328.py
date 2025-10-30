import math

class Solution:
    def minOperations(self, k: int) -> int:
        if k <= 1:
            return 0
        min_ops = float('inf')
        for d in range(k):
            x = math.ceil(k / (d + 1))
            current_ops = x + d - 1
            min_ops = min(min_ops, current_ops)
        return min_ops