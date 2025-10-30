class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        x = n
        while True:
            product = 1
            for ch in str(x):
                digit = int(ch)
                product *= digit
            if product % t == 0:
                return x
            x += 1