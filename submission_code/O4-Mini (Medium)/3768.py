class Solution:
    def hasSameDigits(self, s: str) -> bool:
        # Continue reducing until only two digits remain
        while len(s) > 2:
            # Build the next string by summing pairs of consecutive digits mod 10
            s = ''.join(
                str((int(s[i]) + int(s[i + 1])) % 10) 
                for i in range(len(s) - 1)
            )
        # Check if the final two digits are the same
        return s[0] == s[1]