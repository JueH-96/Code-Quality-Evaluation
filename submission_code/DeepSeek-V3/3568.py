class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        # Pad the numbers with leading zeros to make them 4 digits
        num1_str = f"{num1:04d}"
        num2_str = f"{num2:04d}"
        num3_str = f"{num3:04d}"
        
        # Initialize the key as an empty string
        key = ""
        
        # Iterate over each digit position
        for i in range(4):
            # Get the current digit from each number
            digit1 = int(num1_str[i])
            digit2 = int(num2_str[i])
            digit3 = int(num3_str[i])
            
            # Find the minimum digit
            min_digit = min(digit1, digit2, digit3)
            
            # Append the minimum digit to the key
            key += str(min_digit)
        
        # Convert the key to an integer to remove leading zeros
        return int(key)