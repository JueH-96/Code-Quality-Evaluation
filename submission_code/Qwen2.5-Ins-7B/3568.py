class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        num1_str = str(num1).zfill(4)
        num2_str = str(num2).zfill(4)
        num3_str = str(num3).zfill(4)
        
        key = ""
        for i in range(4):
            key += str(min(int(num1_str[i]), int(num2_str[i]), int(num3_str[i])))
        
        return int(key)