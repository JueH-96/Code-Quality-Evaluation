class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        num1 = str(num1).zfill(4)
        num2 = str(num2).zfill(4)
        num3 = str(num3).zfill(4)
        
        key = int(min(num1[0], num2[0], num3[0])) * 1000 + \
              int(min(num1[1], num2[1], num3[1])) * 100 + \
              int(min(num1[2], num2[2], num3[2])) * 10 + \
              int(min(num1[3], num2[3], num3[3]))
        
        return key