class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        if num1 == 0:
            return 0
        
        for i in range(61):
            diff = 2 ** i + num2
            if num1 >= diff:
                return 1 + self.makeTheIntegerZero(num1 - diff, num2)
        
        return -1