class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        def product_of_digits(x):
            product = 1
            while x > 0:
                product *= x % 10
                x //= 10
            return product

        k = n
        while True:
            if product_of_digits(k) % t == 0:
                return k
            k += 1