class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        s1 = f"{num1:04d}"
        s2 = f"{num2:04d}"
        s3 = f"{num3:04d}"
        
        key = []
        for i in range(4):
            min_d = min(s1[i], s2[i], s3[i])
            key.append(min_d)
        
        return int(''.join(key))