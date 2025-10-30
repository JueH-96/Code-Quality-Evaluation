import math

class Solution:
    def waysToReachStair(self, k: int) -> int:
        ans = 0
        m = 0
        while True:
            pow2 = 1 << m  # Calculate 2^m
            if pow2 < k:
                m += 1
                continue
            d = pow2 - k
            if d < 0:
                m += 1
                continue
            if d > m + 1:
                m += 1
                continue
            ans += math.comb(m + 1, d)
            m += 1
            # Prevent infinite loop for very large k by breaking after a reasonable upper bound
            if m > 60:
                break
        return ans