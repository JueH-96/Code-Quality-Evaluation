class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        # Convert numbers to strings and pad with leading zeros to ensure they are 4 digits long
        num1_str = str(num1).zfill(4)
        num2_str = str(num2).zfill(4)
        num3_str = str(num3).zfill(4)
        
        # Initialize an empty string to store the key
        key = ""
        
        # Iterate through each digit position (0 to 3)
        for i in range(4):
            # Find the minimum digit at the current position
            min_digit = min(num1_str[i], num2_str[i], num3_str[i])
            # Append the minimum digit to the key
            key += min_digit
        
        # Convert the key back to an integer to remove leading zeros
        return int(key)