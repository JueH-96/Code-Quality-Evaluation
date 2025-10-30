class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        # Convert numbers to strings padded with zeros to 4 characters
        s1 = str(num1).zfill(4)
        s2 = str(num2).zfill(4)
        s3 = str(num3).zfill(4)
        
        key = []
        # For each digit position, find the minimum among the three numbers' corresponding digits
        for i in range(4):
            min_digit = min(s1[i], s2[i], s3[i])
            key.append(min_digit)
        
        # Join the list to form key string and convert it to integer to remove leading zeros
        return int("".join(key))