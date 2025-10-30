class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        def digit_product(x: int) -> int:
            product = 1
            while x > 0:
                product *= x % 10
                x //= 10
            return product
        
        current = n
        while True:
            # Compute the product of digits of the current number
            if digit_product(current) % t == 0:
                return current
            current += 1