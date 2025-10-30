class Solution:
    def smallestString(self, s: str) -> str:
        s = list(s)
        for i in range(len(s)):
            if s[i] != 'a':
                for j in range(i, len(s)):
                    if s[j] == 'a':
                        break
                    s[j] = chr(ord(s[j]) - 1)
                break
            if i == len(s) - 1:
                s[i] = 'z'
        return ''.join(s)