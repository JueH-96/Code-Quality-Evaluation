class Solution:
    def hasSameDigits(self, s: str) -> bool:
        # Repeat the operation until only two digits remain
        while len(s) > 2:
            # Compute the new string by summing consecutive digits mod 10
            new_s = []
            for i in range(len(s) - 1):
                new_s.append(str((int(s[i]) + int(s[i + 1])) % 10))
            s = ''.join(new_s)
        # Check if the final two digits are the same
        return s[0] == s[1]