class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        # Convert numbers to strings and pad with leading zeros if necessary
        num1_str = str(num1).zfill(4)
        num2_str = str(num2).zfill(4)
        num3_str = str(num3).zfill(4)
        
        # Initialize the key as an empty string
        key = ""
        
        # Iterate over each digit position (0 to 3)
        for i in range(4):
            # Find the minimum digit at the current position
            min_digit = min(num1_str[i], num2_str[i], num3_str[i])
            # Append the minimum digit to the key
            key += min_digit
        
        # Convert the key string to an integer and remove leading zeros
        key_int = int(key)
        
        return key_int