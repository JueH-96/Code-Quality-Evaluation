class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        # Convert each number to a 4-digit string with leading zeros
        s1, s2, s3 = str(num1).zfill(4), str(num2).zfill(4), str(num3).zfill(4)
        
        # Compute the key digit by digit.
        key_str = ""
        for i in range(4):
            key_str += str(min(s1[i], s2[i], s3[i]))
        
        # Return the integer representation (this will remove any leading zeros)
        return int(key_str)