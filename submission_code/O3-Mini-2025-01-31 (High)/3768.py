class Solution:
    def hasSameDigits(self, s: str) -> bool:
        # Continue performing the operation until the string has exactly two digits.
        while len(s) > 2:
            new_digits = []
            # Compute the modulo sum for each pair of consecutive digits.
            for i in range(len(s) - 1):
                # Sum the adjacent digits and take modulo 10.
                new_digit = (int(s[i]) + int(s[i+1])) % 10
                new_digits.append(str(new_digit))
            s = ''.join(new_digits)
        # Return True if the final two digits are the same, otherwise False.
        return s[0] == s[1]