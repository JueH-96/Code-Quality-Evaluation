class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        x = n
        while True:
            product = 1
            for d in str(x):
                product *= int(d)
            if product % t == 0:
                return x
            x += 1