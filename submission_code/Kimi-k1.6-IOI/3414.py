import math

class Solution:
    def waysToReachStair(self, k: int) -> int:
        res = 0
        for m in range(61):  # Check up to 2^60 which is more than enough for k up to 1e9
            s = 1 << m  # Calculate 2^m
            if s < k:
                continue
            d = s - k
            if d > m + 1:
                continue
            res += math.comb(m + 1, d)
        return res