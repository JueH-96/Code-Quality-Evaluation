class Solution:
    def hasSameDigits(self, s: str) -> bool:
        # Continue processing until s has exactly two digits.
        while len(s) > 2:
            new_s = ""
            # Calculate the new sequence of digits using the sum modulo 10.
            for i in range(len(s) - 1):
                new_digit = (int(s[i]) + int(s[i+1])) % 10
                new_s += str(new_digit)
            s = new_s
        # Check if the two remaining digits are the same.
        return s[0] == s[1]