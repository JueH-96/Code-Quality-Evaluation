class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        def product_of_digits(num: int) -> int:
            prod = 1
            for char in str(num):
                prod *= int(char)
            return prod
        
        while True:
            # Calculate product of digits
            if product_of_digits(n) % t == 0:
                return n
            n += 1