class Solution:
    def hasSameDigits(self, s: str) -> bool:
        """
        Performs the specified operation repeatedly on the digit string s until
        it has exactly two digits and returns true if these two digits are the same.

        The operation involves calculating the sum modulo 10 of each consecutive pair of digits
        and forming a new string from these results. This repeats until the string length is 2.

        Args:
            s: The input string consisting of digits. Length is guaranteed to be between 3 and 100.

        Returns:
            True if the final two digits of the string are the same, False otherwise.
        """
        
        # Continue the process as long as the string length is greater than 2.
        # Each iteration of the loop reduces the length of the string by 1.
        while len(s) > 2:
            # Create a list to store the digits of the next string.
            # Using a list and then ''.join() is generally more efficient for 
            # building strings in Python than repeated string concatenation.
            next_digits = []
            
            # Iterate through the current string `s` to compute the new digits.
            # The loop runs from the first character up to the second-to-last character,
            # allowing us to access pairs of consecutive characters (s[i], s[i+1]).
            for i in range(len(s) - 1):
                # Convert the characters s[i] and s[i+1] to integers to perform arithmetic.
                # The problem constraints guarantee that s consists only of digits.
                digit1 = int(s[i])
                digit2 = int(s[i+1])
                
                # Calculate the sum of the two digits modulo 10, as per the operation definition.
                new_digit_val = (digit1 + digit2) % 10
                
                # Convert the resulting digit (0-9) back to its string representation 
                # and append it to our list of digits for the next string.
                next_digits.append(str(new_digit_val))
            
            # After processing all consecutive pairs, join the collected digits 
            # in the list to form the new string `s`. This new string will be used
            # in the next iteration of the while loop (if its length is still > 2).
            s = "".join(next_digits)
            # Example: if next_digits was ['1', '2', '3'], s becomes "123".

        # The loop terminates when len(s) is exactly 2.
        # At this point, we need to check if the two digits in the final string `s` are identical.
        # s[0] refers to the first character (digit) and s[1] refers to the second character (digit).
        # We compare them directly as characters.
        return s[0] == s[1]