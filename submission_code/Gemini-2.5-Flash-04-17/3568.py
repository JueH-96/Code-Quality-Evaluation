class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        # Pad each number with leading zeros to ensure four digits
        s1 = f"{num1:04d}"
        s2 = f"{num2:04d}"
        s3 = f"{num3:04d}"

        key_digits = []

        # Iterate through each digit position (0 to 3)
        for i in range(4):
            # Get the i-th digit from each padded string and convert to integer
            digit1 = int(s1[i])
            digit2 = int(s2[i])
            digit3 = int(s3[i])

            # Find the smallest digit among the three at this position
            min_digit = min(digit1, digit2, digit3)

            # Append the minimum digit (as a string) to the key digits list
            key_digits.append(str(min_digit))

        # Join the list of digit strings to form the key string
        key_str = "".join(key_digits)

        # Convert the key string to an integer. This automatically handles
        # removing leading zeros if they exist (e.g., "0000" becomes 0,
        # "0777" becomes 777).
        return int(key_str)