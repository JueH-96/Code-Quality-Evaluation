class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        def product_of_digits(x):
            product = 1
            while x > 0:
                digit = x % 10
                product *= digit
                x = x // 10
            return product
        
        current = n
        while True:
            if product_of_digits(current) % t == 0:
                return current
            current += 1