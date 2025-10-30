class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        def digit_product(x):
            product = 1
            while x > 0:
                product *= x % 10
                x = x // 10
            return product
        
        current = n
        while True:
            if digit_product(current) % t == 0:
                return current
            current += 1