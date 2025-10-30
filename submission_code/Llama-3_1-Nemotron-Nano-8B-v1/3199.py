class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        def comb(m):
            if m < 2:
                return 0
            return m * (m - 1) // 2
        
        term0 = comb(n + 2)
        term1 = 3 * comb(n - (limit + 1) + 2)
        term2 = 3 * comb(n - 2 * (limit + 1) + 2)
        term3 = comb(n - 3 * (limit + 1) + 2)
        
        total = term0 - term1 + term2 - term3
        return max(total, 0)