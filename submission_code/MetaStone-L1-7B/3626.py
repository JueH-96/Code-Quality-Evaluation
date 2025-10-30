class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        for num in range(n, 10**6):  # A sufficiently large upper bound
            s = str(num)
            if '0' in s:
                return num
            product = 1
            for c in s:
                digit = int(c)
                product *= digit
            if product % t == 0:
                return num
        return num  # This line is theoretically unreachable