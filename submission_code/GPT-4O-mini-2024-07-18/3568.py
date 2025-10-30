class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        # Pad the numbers with leading zeros to ensure they are 4 digits
        num1_str = str(num1).zfill(4)
        num2_str = str(num2).zfill(4)
        num3_str = str(num3).zfill(4)
        
        # Generate the key by taking the minimum digit at each position
        key_digits = []
        for i in range(4):
            min_digit = min(num1_str[i], num2_str[i], num3_str[i])
            key_digits.append(min_digit)
        
        # Join the digits to form the key and convert to int to remove leading zeros
        key = int(''.join(key_digits))
        return key