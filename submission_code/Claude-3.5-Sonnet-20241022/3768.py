class Solution:
    def hasSameDigits(self, s: str) -> bool:
        # Continue processing until we get exactly 2 digits
        while len(s) > 2:
            # Create new string from pairs of consecutive digits
            new_s = ""
            for i in range(len(s) - 1):
                # Calculate sum modulo 10 for each pair
                digit_sum = (int(s[i]) + int(s[i + 1])) % 10
                new_s += str(digit_sum)
            s = new_s
        
        # Check if final two digits are the same
        return s[0] == s[1]