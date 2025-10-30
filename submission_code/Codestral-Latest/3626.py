class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        def digit_product(num):
            product = 1
            for digit in str(num):
                product *= int(digit)
            return product

        for i in range(n, 101):
            if digit_product(i) % t == 0:
                return i

        return -1  # This line will never be reached due to the constraints