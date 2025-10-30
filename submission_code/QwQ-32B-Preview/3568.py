class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        # Pad the numbers with leading zeros to make them four digits
        num1_str = f"{num1:04d}"
        num2_str = f"{num2:04d}"
        num3_str = f"{num3:04d}"
        
        # Initialize the key string
        key = ""
        
        # Iterate through each digit position
        for i in range(4):
            # Find the smallest digit at position i among the three numbers
            digit = min(num1_str[i], num2_str[i], num3_str[i])
            # Append the smallest digit to the key string
            key += digit
        
        # Convert the key string to an integer and return
        return int(key)