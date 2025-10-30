class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        while True:
            product = 1
            num_str = str(n)
            for digit in num_str:
                product *= int(digit)
            if product % t == 0:
                return n
            n += 1