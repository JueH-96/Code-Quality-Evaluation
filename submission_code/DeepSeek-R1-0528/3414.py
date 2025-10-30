import math

class Solution:
    def waysToReachStair(self, k: int) -> int:
        ans = 0
        n = 0
        while n <= 100:
            current_stair_after_ups = 1 << n
            m = current_stair_after_ups - k
            if m < 0:
                n += 1
                continue
            if m <= n + 1:
                ans += math.comb(n + 1, m)
            if current_stair_after_ups > k and m > n + 1:
                break
            n += 1
        return ans