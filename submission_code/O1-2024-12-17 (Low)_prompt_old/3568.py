class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        # Convert the input numbers to 4-digit strings with leading zeros
        s1 = f"{num1:04d}"
        s2 = f"{num2:04d}"
        s3 = f"{num3:04d}"
        
        # For each digit (index), take the smallest digit from the three numbers
        key_digits = []
        for i in range(4):
            min_digit = min(s1[i], s2[i], s3[i])
            key_digits.append(min_digit)
        
        # Combine the digits to form the 4-digit key string
        key_str = "".join(key_digits)
        
        # Convert to integer to remove leading zeros
        key_int = int(key_str)
        
        return key_int