class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        for k in range(1, 61):
            diff = num1 - k * num2
            if diff < 0:
                continue
            set_bits = bin(diff).count('1')
            if set_bits <= k:
                return k
        return -1