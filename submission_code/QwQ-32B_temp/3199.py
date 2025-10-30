class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        total = 0
        coeffs = [1, -3, 3, -1]
        for k in range(4):
            m = n - k * (limit + 1)
            if m < 0:
                term = 0
            else:
                term = (m + 2) * (m + 1) // 2
            total += coeffs[k] * term
        return total