import math

class Solution:
    def minOperations(self, k: int) -> int:
        if k <= 1:
            return 0
        x_floor = math.isqrt(k)
        x_ceil = x_floor + 1
        f_xf = x_floor + (k + x_floor - 1) // x_floor - 2
        f_xc = x_ceil + (k + x_ceil - 1) // x_ceil - 2
        return min(f_xf, f_xc)