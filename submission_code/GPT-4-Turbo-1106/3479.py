class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        count = 0
        ones = 0
        zeros = 0
        for i in range(len(s)):
            if s[i] == '0':
                zeros += 1
                ones = 0  # Reset ones count after encountering a zero
            else:
                ones += 1
                # Add all substrings ending with the current '1' that have dominant ones
                count += ones - zeros**2 + 1 if zeros == 0 else ones - zeros**2
        return count