class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        def digit_product(x: int) -> int:
            prod = 1
            for ch in str(x):
                prod *= int(ch)
            return prod

        i = n
        while True:
            if digit_product(i) % t == 0:
                return i
            i += 1