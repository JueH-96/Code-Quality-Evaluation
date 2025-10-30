import math

class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        res = 0
        for k in range(4):
            sign = (-1) ** k
            c = math.comb(3, k)
            current_m = n - k * (limit + 1)
            if current_m < 0:
                t = 0
            else:
                t = (current_m + 2) * (current_m + 1) // 2
            res += sign * c * t
        return res