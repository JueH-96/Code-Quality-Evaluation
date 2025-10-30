class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def price(num):
            return sum((num >> (i * x)) & 1 for i in range(64))
        
        def total_price(n):
            return sum(price(i) for i in range(1, n + 1))
        
        l, r = 0, 2**64
        while l < r:
            m = (l + r) // 2
            if total_price(m) <= k:
                l = m + 1
            else:
                r = m
        return l - 1