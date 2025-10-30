class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        for k in range(1, 61):
            target = num1 - k * num2
            if target < k or target < 0:
                continue
            bits = bin(target).count('1')
            if bits <= k:
                return k
        return -1