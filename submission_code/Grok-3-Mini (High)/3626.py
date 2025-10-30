class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        num = n
        while True:
            prod = 1
            for char in str(num):
                digit = int(char)
                prod *= digit
            if prod % t == 0:
                return num
            num += 1