class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        # Convert numbers to strings with leading zeros
        num1_str = str(num1).zfill(4)
        num2_str = str(num2).zfill(4)
        num3_str = str(num3).zfill(4)
        
        # Generate the key by taking the smallest digit at each position
        key = ''.join(min(a, b, c) for a, b, c in zip(num1_str, num2_str, num3_str))
        
        # Convert the key back to an integer to remove leading zeros
        return int(key)