import math

class Solution:
    def waysToReachStair(self, k: int) -> int:
        ans = 0
        for m in range(60):  # 2^60 is a very large number, covering all possible k up to 1e18
            pow2_m = 1 << m  # Calculate 2^m
            lower = pow2_m - m - 1
            upper = pow2_m
            if lower <= k <= upper:
                t = pow2_m - k
                ans += math.comb(m + 1, t)
        return ans