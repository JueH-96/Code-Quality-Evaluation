class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        m = n
        while True:
            prod = 1
            for ch in str(m):
                prod *= int(ch)
            if prod % t == 0:
                return m
            m += 1