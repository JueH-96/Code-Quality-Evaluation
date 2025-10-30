class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        m = n
        while True:
            digits = list(map(int, str(m)))
            if 0 in digits:
                return m
            product = 1
            for d in digits:
                product *= d
            if product % t == 0:
                return m
            m += 1