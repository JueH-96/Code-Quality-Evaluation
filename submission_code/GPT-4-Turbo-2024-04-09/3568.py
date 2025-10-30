class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        # Convert numbers to zero-padded strings
        s1 = f"{num1:04d}"
        s2 = f"{num2:04d}"
        s3 = f"{num3:04d}"
        
        # Generate the key by taking the minimum of each corresponding digit
        key = ""
        for i in range(4):
            min_digit = min(s1[i], s2[i], s3[i])
            key += min_digit
        
        # Convert the key back to integer to remove leading zeros
        return int(key)