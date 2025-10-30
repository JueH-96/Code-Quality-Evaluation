class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        def product_of_digits(num):
            product = 1
            for digit in str(num):
                product *= int(digit)
            return product
        
        current = n
        while True:
            prod = product_of_digits(current)
            if prod % t == 0:
                return current
            current += 1