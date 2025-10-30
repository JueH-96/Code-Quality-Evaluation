class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        # Convert the numbers to strings and pad them with leading zeros
        str_num1 = str(num1).zfill(4)
        str_num2 = str(num2).zfill(4)
        str_num3 = str(num3).zfill(4)
        
        # Initialize an empty string to store the key
        key = ""
        
        # Iterate over the digits of the padded strings
        for i in range(4):
            # Find the minimum digit among the i-th digits of the three numbers
            min_digit = min(str_num1[i], str_num2[i], str_num3[i])
            # Append the minimum digit to the key
            key += min_digit
        
        # Convert the key to an integer
        key_int = int(key)
        
        # Return the key without leading zeros
        return key_int