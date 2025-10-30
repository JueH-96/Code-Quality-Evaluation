class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        # Convert numbers to strings and pad with leading zeros to ensure they are 4 digits
        str_num1 = str(num1).zfill(4)
        str_num2 = str(num2).zfill(4)
        str_num3 = str(num3).zfill(4)
        
        # Initialize the key as an empty string
        key = ""
        
        # Iterate through each digit position
        for i in range(4):
            # Find the minimum digit at the current position
            min_digit = min(str_num1[i], str_num2[i], str_num3[i])
            # Append the minimum digit to the key
            key += min_digit
        
        # Convert the key to an integer to remove any leading zeros
        return int(key)