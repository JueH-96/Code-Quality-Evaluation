class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        # Convert the numbers to strings and pad them with leading zeros
        num1_str = str(num1).zfill(4)
        num2_str = str(num2).zfill(4)
        num3_str = str(num3).zfill(4)

        # Initialize an empty string to store the key
        key_str = ""

        # Iterate over the digits of the numbers
        for i in range(4):
            # Find the smallest digit among the i^th digits of num1, num2, and num3
            min_digit = min(int(num1_str[i]), int(num2_str[i]), int(num3_str[i]))
            # Append the smallest digit to the key
            key_str += str(min_digit)

        # Convert the key string to an integer and return it
        return int(key_str)