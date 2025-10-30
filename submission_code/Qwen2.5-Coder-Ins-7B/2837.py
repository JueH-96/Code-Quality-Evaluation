class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        for i in range(61):
            if num1 - i * num2 <= 0 and bin(num1 - i * num2).count('1') <= i and i <= num1 - i * num2:
                return i
        return -1