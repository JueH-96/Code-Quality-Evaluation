class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        for k in range(1, 61):
            x = num1 - k * num2
            if x < 0 or x < k:
                continue
            m = bin(x).count('1')
            if m <= k:
                return k
        return -1