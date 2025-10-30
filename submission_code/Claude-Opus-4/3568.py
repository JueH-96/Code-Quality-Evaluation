class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        # Convert numbers to 4-digit strings with leading zeros
        str1 = str(num1).zfill(4)
        str2 = str(num2).zfill(4)
        str3 = str(num3).zfill(4)
        
        # Build the key by taking minimum digit at each position
        key = ""
        for i in range(4):
            min_digit = min(str1[i], str2[i], str3[i])
            key += min_digit
        
        # Convert to integer to remove leading zeros
        return int(key)