class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        # Convert each number to a 4-digit string with leading zeros
        s1 = f"{num1:04d}"
        s2 = f"{num2:04d}"
        s3 = f"{num3:04d}"
        
        # Build the key by taking the minimum digit at each position
        key_digits = []
        for i in range(4):
            min_digit = min(int(s1[i]), int(s2[i]), int(s3[i]))
            key_digits.append(str(min_digit))
        
        # Join the digits and convert to int (to remove any leading zeros)
        return int("".join(key_digits))