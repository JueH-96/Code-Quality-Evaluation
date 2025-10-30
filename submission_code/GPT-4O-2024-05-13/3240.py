class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def price(num, x):
            s = bin(num)[2:]
            return sum(1 for i in range(len(s)) if (i + 1) % x == 0 and s[-(i + 1)] == '1')
        
        total_price = 0
        num = 0
        
        while total_price <= k:
            num += 1
            total_price += price(num, x)
        
        return num - 1