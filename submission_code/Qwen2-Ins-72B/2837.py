class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        for k in range(1, 61):
            x = num1 - k * num2
            if 0 <= x < (1 << k) and x.bit_count() <= k:
                return k
        return -1