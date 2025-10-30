class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        # Convert numbers to strings and pad with leading zeros to make them four digits
        num1_str = str(num1).zfill(4)
        num2_str = str(num2).zfill(4)
        num3_str = str(num3).zfill(4)
        
        key_str = ""
        
        # Iterate through each of the four positions
        for i in range(4):
            # Get the digits at position i
            digit1 = num1_str[i]
            digit2 = num2_str[i]
            digit3 = num3_str[i]
            # Find the smallest digit
            min_digit = str(min(digit1, digit2, digit3))
            # Append the smallest digit to the key string
            key_str += min_digit
        
        # Convert the key string to an integer and return it
        key = int(key_str)
        return key