class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        # Convert the numbers to strings and pad with leading zeros if necessary
        num1_str = str(num1).zfill(4)
        num2_str = str(num2).zfill(4)
        num3_str = str(num3).zfill(4)
        
        # Initialize the key as an empty string
        key = ""
        
        # Iterate through the digits of the numbers and find the smallest digit at each position
        for i in range(4):
            key += str(min(int(num1_str[i]), int(num2_str[i]), int(num3_str[i])))
        
        # Remove leading zeros from the key
        key = str(int(key))
        
        return int(key)