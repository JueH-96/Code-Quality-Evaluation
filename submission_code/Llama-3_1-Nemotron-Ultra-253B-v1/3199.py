import math

class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        def comb(n, k):
            if n < 0 or k < 0 or n < k:
                return 0
            return math.comb(n, k)
        
        t = comb(n + 2, 2)
        a1 = comb(n - (limit + 1) + 2, 2)
        a2 = comb(n - 2 * (limit + 1) + 2, 2)
        a3 = comb(n - 3 * (limit + 1) + 2, 2)
        return t - 3 * a1 + 3 * a2 - a3