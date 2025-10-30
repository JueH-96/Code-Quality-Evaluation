class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        def comb(m):
            if m < 0:
                return 0
            return (m + 2) * (m + 1) // 2
        
        total = comb(n)
        a = comb(n - (limit + 1))
        ab = comb(n - 2 * (limit + 1))
        abc_term = comb(n - 3 * (limit + 1))
        
        res = total - 3 * a + 3 * ab - abc_term
        return max(res, 0)