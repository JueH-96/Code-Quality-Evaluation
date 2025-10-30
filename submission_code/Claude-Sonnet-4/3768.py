class Solution:
    def hasSameDigits(self, s: str) -> bool:
        # Continue until we have exactly 2 digits
        while len(s) > 2:
            new_digits = []
            # For each pair of consecutive digits
            for i in range(len(s) - 1):
                # Calculate sum modulo 10
                digit_sum = (int(s[i]) + int(s[i + 1])) % 10
                new_digits.append(str(digit_sum))
            # Update s with the new sequence
            s = ''.join(new_digits)
        
        # Check if the final two digits are the same
        return s[0] == s[1]