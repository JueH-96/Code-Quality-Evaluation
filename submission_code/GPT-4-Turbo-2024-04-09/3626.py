class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        def product_of_digits(x):
            product = 1
            while x > 0:
                digit = x % 10
                if digit == 0:
                    return 0
                product *= digit
                x //= 10
            return product
        
        while True:
            if product_of_digits(n) % t == 0:
                return n
            n += 1