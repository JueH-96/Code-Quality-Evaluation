class Solution:
    def hasSameDigits(self, s: str) -> bool:
        """
        Repeatedly performs an operation on a string of digits until it has
        exactly two digits. The operation is to replace the string with a
        new string where each digit is the sum of consecutive digits in
        the original string, modulo 10.

        Finally, checks if the two resulting digits are the same.

        Args:
            s: A string consisting of digits.

        Returns:
            True if the final two digits are the same, False otherwise.
        """
        current_s = s

        # Continue the operation as long as the string length is greater than 2
        while len(current_s) > 2:
            next_s_list = []
            # Iterate through the string to process consecutive pairs
            # The loop goes up to len(current_s) - 1 because we access i and i+1
            for i in range(len(current_s) - 1):
                # Convert digit characters to integers
                digit1 = int(current_s[i])
                digit2 = int(current_s[i+1])

                # Calculate the sum modulo 10
                new_digit_int = (digit1 + digit2) % 10

                # Convert the result back to a string digit and append to the list
                next_s_list.append(str(new_digit_int))

            # Join the list of digits to form the new string for the next iteration
            current_s = "".join(next_s_list)

        # After the loop, current_s has exactly two digits.
        # Check if the two digits are the same.
        return current_s[0] == current_s[1]