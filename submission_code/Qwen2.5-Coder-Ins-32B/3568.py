class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        # Convert numbers to strings and pad with leading zeros to make them 4 digits
        num1_str = f"{num1:04}"
        num2_str = f"{num2:04}"
        num3_str = f"{num3:04}"
        
        # Initialize the key as an empty string
        key = ""
        
        # Iterate over each digit position
        for i in range(4):
            # Find the minimum digit at the current position
            min_digit = min(num1_str[i], num2_str[i], num3_str[i])
            # Append the minimum digit to the key
            key += min_digit
        
        # Convert the key to an integer to remove leading zeros
        return int(key)