class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        # Convert numbers to strings and pad with leading zeros to ensure they are 4 digits long
        str1 = str(num1).zfill(4)
        str2 = str(num2).zfill(4)
        str3 = str(num3).zfill(4)

        # Initialize an empty list to store the digits of the key
        key_digits = []

        # Iterate over each digit position (0 to 3)
        for i in range(4):
            # Find the minimum digit at the i-th position among the three numbers
            min_digit = min(str1[i], str2[i], str3[i])
            # Append the minimum digit to the key_digits list
            key_digits.append(min_digit)

        # Join the list of digits to form the key as a string
        key_str = ''.join(key_digits)

        # Convert the key string to an integer and return it
        return int(key_str)