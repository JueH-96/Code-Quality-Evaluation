class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        # Convert to 4-digit strings with leading zeros
        str1 = f"{num1:04d}"
        str2 = f"{num2:04d}"
        str3 = f"{num3:04d}"
        
        key = ""
        for i in range(4):
            digit1 = int(str1[i])
            digit2 = int(str2[i])
            digit3 = int(str3[i])
            min_digit = min(digit1, digit2, digit3)
            key += str(min_digit)
        
        return int(key)