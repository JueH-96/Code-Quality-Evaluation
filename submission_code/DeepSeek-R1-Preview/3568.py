class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        s1 = str(num1).zfill(4)
        s2 = str(num2).zfill(4)
        s3 = str(num3).zfill(4)
        
        key = []
        for i in range(4):
            d1 = s1[i]
            d2 = s2[i]
            d3 = s3[i]
            min_d = min(d1, d2, d3)
            key.append(min_d)
        
        return int(''.join(key))