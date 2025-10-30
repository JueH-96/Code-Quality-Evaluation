class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        if num2 < 0:
            num1 += abs(num2)
            num2 = abs(num2)
        operations = 0
        while num1 > 0:
            i = 0
            while (2 ** (i + 1)) <= num1:
                i += 1
            num1 -= (2 ** i + num2)
            operations += 1
            if num1 < 0:
                return -1
        return operations