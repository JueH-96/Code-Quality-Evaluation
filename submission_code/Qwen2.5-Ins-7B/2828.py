class Solution:
    def smallestString(self, s: str) -> str:
        n = len(s)
        i = 0
        while i < n and s[i] == 'a':
            i += 1
        if i == n:
            return s[:-1] + 'z'
        while i < n and s[i] != 'a':
            s = s[:i] + chr(ord(s[i]) - 1) + s[i+1:]
            i += 1
        return s