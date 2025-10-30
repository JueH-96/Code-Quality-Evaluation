import math

class Solution:
    def waysToReachStair(self, k: int) -> int:
        total = 0
        m = 0
        # Iterate up to a reasonable limit since 2^60 is very large
        while m <= 60:
            pow2m = 1 << m  # Compute 2^m efficiently
            if pow2m < k:
                m += 1
                continue
            d = pow2m - k
            if d < 0 or d > m + 1:
                m += 1
                continue
            # Add the number of ways to choose d slots from m+1
            total += math.comb(m + 1, d)
            m += 1
        return total