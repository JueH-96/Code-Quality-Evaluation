class Solution:
    def smallestString(self, s: str) -> str:
        chars = list(s)
        n = len(chars)
        i = 0
        shifted = False
        while i < n:
            if chars[i] > 'a':
                j = i
                while j < n and chars[j] > 'a':
                    j += 1
                for k in range(i, j):
                    chars[k] = chr(ord(chars[k]) - 1)
                shifted = True
                break
            i += 1
        if not shifted:
            # Shift the last character from 'a' to 'z'
            chars[-1] = 'z'
        return ''.join(chars)