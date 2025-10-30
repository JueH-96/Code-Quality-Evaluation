class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        def product(num):
            prod = 1
            if num == 0:
                return 0
            while num > 0:
                prod *= num % 10
                num //= 10
            return prod

        i = n
        while True:
            if product(i) % t == 0:
                return i
            i += 1