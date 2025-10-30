class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        for i in range(61, -1, -1):
            if num1 - 2**i - num2 >= 0 and num1 - 2**i - num2 >= num1 - 2**(i-1) - num2:
                return i
        return -1