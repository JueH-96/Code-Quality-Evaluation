class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        # Convert numbers to zero-padded 4-digit strings
        s1 = str(num1).zfill(4)
        s2 = str(num2).zfill(4)
        s3 = str(num3).zfill(4)
        
        # Form the key by taking the min digit at each position
        key_digits = []
        for i in range(4):
            key_digits.append(min(s1[i], s2[i], s3[i]))
        
        # Convert the 4-digit key to an integer to strip leading zeros
        key_str = "".join(key_digits)
        return int(key_str)