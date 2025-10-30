class Solution:
    def waysToReachStair(self, k: int) -> int:
        def comb(n, k):
            if k < 0 or k > n:
                return 0
            if k == 0 or k == n:
                return 1
            k = min(k, n - k)  # Take advantage of symmetry
            result = 1
            for i in range(1, k + 1):
                result = result * (n - i + 1) // i
            return result
        
        total = 0
        max_m = 40  # Sufficiently large to cover 2^40 which is over 1e12
        for m in range(0, max_m + 1):
            S = 2 ** m
            d = S - k
            if d < 0:
                continue
            if d > m + 1:
                continue
            total += comb(m + 1, d)
        return total