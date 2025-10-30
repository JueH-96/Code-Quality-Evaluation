class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        def comb(n,k):
            if k < 0 or k > n:
                return 0
            res = 1
            for i in range(1, k+1):
                res = res * (n - i + 1) // i
            return res
        
        total = comb(n + 2, 2)
        res = total
        for mask in range(1, 8):
            bits = bin(mask).count('1')
            sign = (-1) ** bits
            total_limit = bits * (limit + 1)
            n_remain = n - total_limit
            if n_remain >= 0:
                cnt = comb(n_remain + 2, 2)
                res += sign * cnt
        return res