class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        # Pad the numbers with leading zeros to ensure they are 4 digits long
        num1_str = f"{num1:04}"
        num2_str = f"{num2:04}"
        num3_str = f"{num3:04}"

        # Initialize the key as an empty string
        key = ""

        # Iterate over each digit position
        for i in range(4):
            # Find the minimum digit at the i-th position
            min_digit = min(num1_str[i], num2_str[i], num3_str[i])
            # Append the minimum digit to the key
            key += min_digit

        # Convert the key to an integer and return it
        return int(key)