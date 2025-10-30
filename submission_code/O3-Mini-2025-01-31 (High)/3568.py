class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        # Pad each number with leading zeros to ensure they are 4-digit strings
        s1 = str(num1).zfill(4)
        s2 = str(num2).zfill(4)
        s3 = str(num3).zfill(4)

        # For each position, select the minimum digit among the three
        key_digits = []
        for d1, d2, d3 in zip(s1, s2, s3):
            key_digits.append(min(d1, d2, d3))
        
        # Join the digits to form the key and convert to int to remove leading zeros
        key_number = int("".join(key_digits))
        return key_number