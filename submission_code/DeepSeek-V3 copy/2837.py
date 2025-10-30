class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        for k in range(1, 61):
            target = num1 - k * num2
            if target < 0:
                continue
            if bin(target).count('1') <= k and target >= k:
                return k
        return -1