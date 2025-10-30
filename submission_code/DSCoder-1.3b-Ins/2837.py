class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        if num1 < num2:
            return -1
        count = 0
        while num1 > 0:
            if num1 >= num2:
                num1 -= num2
            else:
                num1 -= 2**i
            i += 1
            count += 1
        return count