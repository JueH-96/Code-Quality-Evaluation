class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        """
        Generates the key based on the minimum digits at each position after padding the input numbers.

        Args:
            num1: The first positive integer (1 <= num1 <= 9999).
            num2: The second positive integer (1 <= num2 <= 9999).
            num3: The third positive integer (1 <= num3 <= 9999).

        Returns:
            The generated key as an integer, without leading zeros.
        """
        
        # Convert numbers to strings and pad them with leading zeros to ensure they are 4 digits long.
        # The zfill(4) method adds leading zeros if the string length is less than 4.
        s1 = str(num1).zfill(4)
        s2 = str(num2).zfill(4)
        s3 = str(num3).zfill(4)

        key_str = "" # Initialize an empty string to build the key digit by digit

        # Iterate through the four digit positions (index 0 corresponds to the thousands place, 
        # index 1 to hundreds, index 2 to tens, and index 3 to units).
        for i in range(4):
            # Extract the digit character at the current position 'i' from each padded string.
            digit_char1 = s1[i]
            digit_char2 = s2[i]
            digit_char3 = s3[i]
            
            # Convert these digit characters ('0'-'9') to integers (0-9) to perform numerical comparison.
            d1 = int(digit_char1)
            d2 = int(digit_char2)
            d3 = int(digit_char3)

            # Find the smallest digit among the three for the current position 'i'.
            # This smallest digit will become the digit at position 'i' in the final key.
            min_digit = min(d1, d2, d3)

            # Convert the smallest digit (which is an integer) back to its string representation
            # and append it to the key string being built.
            key_str += str(min_digit)

        # After iterating through all four positions, key_str holds the 4-digit key as a string
        # (e.g., "0777", "0001", "0000", "1234").
        # Convert this resulting key string to an integer. The int() function in Python
        # automatically handles the removal of any leading zeros. For example:
        # int("0777") becomes 777
        # int("0001") becomes 1
        # int("0000") becomes 0
        # int("1234") becomes 1234
        key_int = int(key_str)

        # Return the final key as an integer.
        return key_int