class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        def digit_product(num):
            prod = 1
            for ch in str(num):
                prod *= int(ch)
            return prod
        
        while True:
            # Compute the product of the digits of n
            if digit_product(n) % t == 0:
                return n
            n += 1