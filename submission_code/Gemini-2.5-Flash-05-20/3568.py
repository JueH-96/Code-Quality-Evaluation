class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        # Step 1: Convert each number to a 4-digit string,
        # padding with leading zeros if necessary.
        # For example, 1 becomes "0001", 10 becomes "0010", 1000 remains "1000".
        s1 = f"{num1:04d}"
        s2 = f"{num2:04d}"
        s3 = f"{num3:04d}"

        # Initialize an empty list to store the digits of the generated key.
        key_digits_list = []

        # Step 2: Iterate through each digit position from 1st to 4th (index 0 to 3).
        for i in range(4):
            # Extract the digit at the current position 'i' from each string.
            # Convert these character digits to integers for comparison.
            digit1 = int(s1[i])
            digit2 = int(s2[i])
            digit3 = int(s3[i])

            # Find the smallest digit among the three at the current position.
            min_digit_at_pos = min(digit1, digit2, digit3)

            # Convert the smallest digit back to a string and append it to our list.
            key_digits_list.append(str(min_digit_at_pos))

        # Step 3: Join the list of digit strings to form the final key as a string.
        # Example: ["0", "0", "0", "0"] becomes "0000"
        # Example: ["0", "7", "7", "7"] becomes "0777"
        key_str = "".join(key_digits_list)

        # Step 4: Convert the key string to an integer.
        # Python's int() function automatically handles leading zeros,
        # so "0000" becomes 0, "0012" becomes 12, "0777" becomes 777.
        return int(key_str)