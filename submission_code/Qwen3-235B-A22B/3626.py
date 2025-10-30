class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        def product_digits(num):
            product = 1
            while num > 0:
                digit = num % 10
                product *= digit
                num = num // 10
            return product
        
        current = n
        while True:
            if product_digits(current) % t == 0:
                return current
            current += 1