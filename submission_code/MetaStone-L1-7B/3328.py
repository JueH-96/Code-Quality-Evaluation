import math

class Solution:
    def minOperations(self, k: int) -> int:
        if k == 0:
            return 0
        min_f = float('inf')
        sqrt_k = int(math.sqrt(k))
        for x in range(1, sqrt_k + 2):
            d = (k + x - 1) // x
            f = x + d
            if f < min_f:
                min_f = f
        return max(0, min_f - 2)