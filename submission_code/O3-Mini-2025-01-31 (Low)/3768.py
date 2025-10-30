class Solution:
    def hasSameDigits(self, s: str) -> bool:
        # Loop until we have exactly two digits in the string
        while len(s) > 2:
            new_digits = []
            # Iterate over consecutive pairs in the string
            for i in range(len(s) - 1):
                # Compute the sum modulo 10 of the two digits and convert to string
                new_digit = (int(s[i]) + int(s[i + 1])) % 10
                new_digits.append(str(new_digit))
            # New string becomes the joined new digits
            s = "".join(new_digits)
        
        # Check if both digits in the final string are identical
        return s[0] == s[1]