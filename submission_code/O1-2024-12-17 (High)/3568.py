class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        # Convert each number to a zero-padded 4-digit string
        s1 = str(num1).zfill(4)
        s2 = str(num2).zfill(4)
        s3 = str(num3).zfill(4)
        
        # Build the result by taking the minimum digit in each position
        result_digits = [
            min(s1[i], s2[i], s3[i]) for i in range(4)
        ]
        
        # Convert the resulting 4-digit string to int (removing leading zeros)
        return int("".join(result_digits))