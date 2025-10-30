class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        s1 = str(num1).zfill(4)
        s2 = str(num2).zfill(4)
        s3 = str(num3).zfill(4)
        
        key = ""
        for i in range(4):
            key += str(min(int(s1[i]), int(s2[i]), int(s3[i])))
        
        return int(key)