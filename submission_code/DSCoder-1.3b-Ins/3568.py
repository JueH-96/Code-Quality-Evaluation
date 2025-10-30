class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        num1 = str(num1)
        num2 = str(num2)
        num3 = str(num3)
        
        key = [min(num1[i], num2[i], num3[i]) for i in range(4)]
        return int(''.join(key))