class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        for num in range(n, 101):
            product = 1
            for digit in str(num):
                product *= int(digit)
            if product % t == 0:
                return num
        return -1