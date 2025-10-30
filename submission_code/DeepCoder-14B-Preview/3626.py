class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        current = n
        while True:
            s = str(current)
            digits = [int(c) for c in s]
            if 0 in digits:
                return current
            product = 1
            for d in digits:
                product *= d
            if product % t == 0:
                return current
            current += 1