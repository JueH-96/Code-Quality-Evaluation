class Solution:
    def smallestString(self, s: str) -> str:
        n = len(s)
        i = 0
        # Find the first non 'a' character
        while i < n and s[i] == 'a':
            i += 1
        if i == n:
            # All characters are 'a', shift the entire string
            return 'z' * n if n > 0 else ''
        # Find the first 'a' after i (if any)
        j = i
        while j < n and s[j] != 'a':
            j += 1
        # Shift the substring from i to j-1
        shifted = []
        for c in s[i:j]:
            shifted_c = chr((ord(c) - ord('a') - 1) % 26 + ord('a'))
            shifted.append(shifted_c)
        return s[:i] + ''.join(shifted) + s[j:]