class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        i = n
        while True:
            s = str(i)
            prod = 1
            for char in s:
                digit = int(char)
                prod *= digit
                if prod == 0:
                    break
            if prod % t == 0:
                return i
            i += 1