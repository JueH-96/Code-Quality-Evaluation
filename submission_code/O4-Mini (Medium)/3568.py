class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        # Pad each number to 4 digits with leading zeros
        s1 = f"{num1:04d}"
        s2 = f"{num2:04d}"
        s3 = f"{num3:04d}"
        
        # For each digit position, pick the smallest digit among the three numbers
        key_digits = []
        for i in range(4):
            # min on single-character strings works since '0'<'1'<...<'9'
            key_digits.append(min(s1[i], s2[i], s3[i]))
        
        # Join into a string and convert to int to drop any leading zeros
        key_str = "".join(key_digits)
        return int(key_str)