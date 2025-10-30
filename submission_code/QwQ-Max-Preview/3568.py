class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        # Convert each number to a 4-digit string with leading zeros
        s1 = f"{num1:04d}"
        s2 = f"{num2:04d}"
        s3 = f"{num3:04d}"
        
        key = []
        for i in range(4):
            # Get the i-th digit from each of the three numbers
            digit1 = s1[i]
            digit2 = s2[i]
            digit3 = s3[i]
            # Append the minimum of the three digits to the key
            key.append(min(digit1, digit2, digit3))
        
        # Join the digits to form the key string and convert to integer
        return int(''.join(key))