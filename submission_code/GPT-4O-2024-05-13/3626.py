class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        def product_of_digits(x):
            product = 1
            for digit in str(x):
                product *= int(digit)
            return product
        
        while True:
            if product_of_digits(n) % t == 0:
                return n
            n += 1