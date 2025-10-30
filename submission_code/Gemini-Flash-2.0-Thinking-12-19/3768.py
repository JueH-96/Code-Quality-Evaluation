class Solution:
    def hasSameDigits(self, s: str) -> bool:
        """
        Repeatedly reduces a string of digits by summing consecutive digits modulo 10
        until only two digits remain. Checks if the final two digits are the same.

        Args:
            s: A string consisting of digits, with length >= 3.

        Returns:
            True if the final two digits are the same, False otherwise.
        """
        # Continue the operation until the string has exactly two digits
        while len(s) > 2:
            next_s_list = []
            # Iterate through the string to compute sums of consecutive digits
            # The loop runs len(s) - 1 times, producing a new string of length len(s) - 1
            for i in range(len(s) - 1):
                # Convert consecutive character digits to integers
                digit1 = int(s[i])
                digit2 = int(s[i+1])
                # Calculate the new digit as the sum modulo 10
                new_digit = (digit1 + digit2) % 10
                # Convert the new digit back to a character digit and append to the list
                next_s_list.append(str(new_digit))
            # Join the list of character digits to form the new string
            s = "".join(next_s_list)

        # After the loop terminates, the string s has length exactly 2
        # Check if the first and second digits are the same
        return s[0] == s[1]