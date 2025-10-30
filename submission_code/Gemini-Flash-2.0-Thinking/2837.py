class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        if num1 == 0:
            return 0

        if num2 > 0:
            for k in range(1, num1 + 2):
                target = num1 - k * num2
                if target < 0:
                    return -1
                if target >= 0 and bin(target).count('1') <= k:
                    return k
            return -1
        else:
            for k in range(1, 65):
                target = num1 - k * num2
                if target >= 0 and bin(target).count('1') <= k:
                    return k
            return -1