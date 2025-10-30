class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        def digit_product(num):
            product = 1
            while num > 0:
                digit = num % 10
                product *= digit
                num //= 10
            return product
        
        current = n
        while True:
            if digit_product(current) % t == 0:
                return current
            current += 1