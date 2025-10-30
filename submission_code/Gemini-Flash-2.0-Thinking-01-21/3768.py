class Solution:
    def hasSameDigits(self, s: str) -> bool:
        # The loop continues as long as the string length is greater than 2.
        while len(s) > 2:
            # Create a new string by processing consecutive digits.
            new_s_list = []
            # Iterate through the string up to the second-to-last character.
            for i in range(len(s) - 1):
                # Convert consecutive digit characters to integers.
                digit1 = int(s[i])
                digit2 = int(s[i+1])
                # Calculate the sum modulo 10.
                new_digit = (digit1 + digit2) % 10
                # Convert the new digit back to a string and append it to the list.
                new_s_list.append(str(new_digit))
            # Join the list of digit strings to form the new string.
            s = "".join(new_s_list)

        # After the loop, the string s will have exactly two digits.
        # Check if the first digit is equal to the second digit.
        return s[0] == s[1]