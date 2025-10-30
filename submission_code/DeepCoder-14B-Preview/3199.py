import math

class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        total = 0
        for k in range(0, 4):
            sign = (-1) ** k
            comb_3k = math.comb(3, k)
            m = n - k * (limit + 1)
            if m < 0:
                continue
            ways = math.comb(m + 2, 2)
            term = sign * comb_3k * ways
            total += term
        return total