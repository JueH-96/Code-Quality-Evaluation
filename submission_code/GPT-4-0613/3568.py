class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        num1, num2, num3 = str(num1).zfill(4), str(num2).zfill(4), str(num3).zfill(4)
        key = ''.join(str(min(int(num1[i]), int(num2[i]), int(num3[i]))) for i in range(4))
        return int(key)