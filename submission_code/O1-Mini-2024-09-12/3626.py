class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        def digit_product(num: int) -> int:
            product = 1
            for digit in str(num):
                product *= int(digit)
            return product

        current = n
        while True:
            product = digit_product(current)
            if product % t == 0:
                return current
            current += 1