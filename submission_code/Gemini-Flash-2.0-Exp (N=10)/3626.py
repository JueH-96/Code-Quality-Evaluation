class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        def digit_product(num):
            prod = 1
            for digit in str(num):
                prod *= int(digit)
            return prod

        while True:
            if digit_product(n) % t == 0:
                return n
            n += 1