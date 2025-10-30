import math

class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        res = 0
        for k in range(0, 4):
            t = n - k * (limit + 1)
            if t >= 0:
                c = (t + 2) * (t + 1) // 2
                res += (-1) ** k * math.comb(3, k) * c
        return res