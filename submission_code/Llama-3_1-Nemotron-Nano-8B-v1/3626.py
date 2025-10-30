class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        m = n
        while True:
            s = str(m)
            if '0' in s:
                return m
            product = 1
            for c in s:
                product *= int(c)
            if product % t == 0:
                return m
            m += 1