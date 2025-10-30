class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        # Pad each number to 4 digits with leading zeros
        s1 = f"{num1:04d}"
        s2 = f"{num2:04d}"
        s3 = f"{num3:04d}"
        
        # For each digit position, take the smallest digit among the three numbers
        key_str = ''.join(
            min(c1, c2, c3) 
            for c1, c2, c3 in zip(s1, s2, s3)
        )
        
        # Convert to int to strip leading zeros (int("0000") -> 0)
        return int(key_str)