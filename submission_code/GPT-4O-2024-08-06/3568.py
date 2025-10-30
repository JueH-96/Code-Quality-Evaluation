class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        # Convert numbers to strings and pad with leading zeros to make them 4 digits
        str1 = f"{num1:04}"
        str2 = f"{num2:04}"
        str3 = f"{num3:04}"
        
        # Initialize an empty list to store the digits of the key
        key_digits = []
        
        # Iterate over each digit position (0 to 3 for 4 digits)
        for i in range(4):
            # Find the minimum digit at the current position among the three numbers
            min_digit = min(str1[i], str2[i], str3[i])
            # Append the minimum digit to the key_digits list
            key_digits.append(min_digit)
        
        # Join the key_digits list into a string and convert it to an integer to remove leading zeros
        key = int(''.join(key_digits))
        
        return key