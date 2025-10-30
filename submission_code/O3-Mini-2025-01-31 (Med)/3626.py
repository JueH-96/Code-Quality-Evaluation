class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        def product_of_digits(num: int) -> int:
            prod = 1
            while num > 0:
                prod *= num % 10
                num //= 10
            return prod
        
        while True:
            if product_of_digits(n) % t == 0:
                return n
            n += 1