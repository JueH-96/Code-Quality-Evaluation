import math

class Solution:
    def waysToReachStair(self, k: int) -> int:
        res = 0
        max_m = 60  # Sufficiently large to cover 2^60 which is way beyond 1e9
        for m in range(0, max_m + 1):
            pow2 = 1 << m  # Calculate 2^m
            d = pow2 - k
            if d >= 0:
                if d <= m + 1:
                    res += math.comb(m + 1, d)
        return res