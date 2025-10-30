class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        candidate = n
        while True:
            product = 1
            for d in str(candidate):
                product *= int(d)
            if product % t == 0:
                return candidate
            candidate += 1