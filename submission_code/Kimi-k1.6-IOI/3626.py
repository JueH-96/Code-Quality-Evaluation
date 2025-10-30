class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        m = n
        while True:
            product = 1
            temp = m
            while temp > 0:
                product *= temp % 10
                temp = temp // 10
            if product % t == 0:
                return m
            m += 1