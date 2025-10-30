class Solution:
    def smallestString(self, s: str) -> str:
        n = len(s)
        i = 0
        # Find the first non-'a' character
        while i < n and s[i] == 'a':
            i += 1
        # If all characters are 'a', transform the entire string
        if i == n:
            return ''.join('z' if c == 'a' else chr(ord(c) - 1) for c in s)
        # Find the maximum j where s[i..j] are all non-'a'
        j = i
        while j < n and s[j] != 'a':
            j += 1
        j -= 1  # adjust to the last non-'a' in the consecutive sequence
        # Transform the substring from i to j
        res = list(s)
        for k in range(i, j + 1):
            res[k] = chr(ord(res[k]) - 1)
        return ''.join(res)