class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        s1 = str(num1).zfill(4)
        s2 = str(num2).zfill(4)
        s3 = str(num3).zfill(4)
        
        res = []
        for i in range(4):
            min_digit = min(s1[i], s2[i], s3[i])
            res.append(min_digit)
            
        key_str = ''.join(res)
        return int(key_str)