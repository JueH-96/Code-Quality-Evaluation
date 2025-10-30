class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        if num1 < abs(num2):
            return -1

        for k in range(1, 61):
            target = num1 - k * num2
            if target >= 0 and target.bit_count() <= k <= target:
                return k
        return -1