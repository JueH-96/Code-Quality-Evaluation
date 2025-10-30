class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        s1 = f"{num1:04d}"
        s2 = f"{num2:04d}"
        s3 = f"{num3:04d}"
        key_digits = []
        for i in range(4):
            key_digits.append(min(s1[i], s2[i], s3[i]))
        return int(''.join(key_digits))