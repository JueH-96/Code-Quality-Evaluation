import math

class Solution:
    def waysToReachStair(self, k: int) -> int:
        ans = 0
        m = 0
        while True:
            pow2 = 1 << m  # Calculate 2^m
            # Check if 2^m exceeds the upper bound for current m
            if pow2 > k + (m + 1):
                break
            lower = pow2 - (m + 1)
            if k >= lower and k <= pow2:
                d = pow2 - k
                if d >= 0 and d <= m + 1:
                    # Add the number of valid sequences for this m and d
                    ans += math.comb(m + 1, d)
            m += 1
        return ans