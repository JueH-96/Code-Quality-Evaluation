class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        current = n
        while True:
            product = 1
            x = current
            while x > 0:
                digit = x % 10
                product *= digit
                x = x // 10
            if product % t == 0:
                return current
            current += 1