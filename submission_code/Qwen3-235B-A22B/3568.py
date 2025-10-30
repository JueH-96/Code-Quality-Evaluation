class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        # Convert each number to 4-digit strings with leading zeros
        str1 = str(num1).zfill(4)
        str2 = str(num2).zfill(4)
        str3 = str(num3).zfill(4)
        
        key_digits = []
        for i in range(4):
            # Get the digits of each number at position i
            d1 = int(str1[i])
            d2 = int(str2[i])
            d3 = int(str3[i])
            # Append the minimum digit as a string
            key_digits.append(str(min(d1, d2, d3)))
        
        # Join the digits to form the key string and convert to integer
        return int(''.join(key_digits))