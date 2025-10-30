class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        if num2 >= 0:
            return -1

        for k in range(1, 61):
            if (num1 - k * num2) >= 0 and (num1 - k * num2).bit_count() <= k:
                return k

        return -1