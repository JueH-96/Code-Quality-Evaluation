class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        # Pad each number with leading zeros to make them four-digit strings
        s1 = str(num1).zfill(4)
        s2 = str(num2).zfill(4)
        s3 = str(num3).zfill(4)

        # Build the key digit by digit taking the minimum of the corresponding digits
        key_digits = [
            min(s1[i], s2[i], s3[i])     # character comparison works for digits ('0' < '1' < ... < '9')
            for i in range(4)
        ]

        key_str = ''.join(key_digits).lstrip('0')  # remove leading zeros
        if key_str == '':
            key_str = '0'                          # if all digits were zero
        
        return int(key_str)