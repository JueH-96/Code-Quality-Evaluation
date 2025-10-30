class Solution:
    def findLatestTime(self, s: str) -> str:
        s = list(s)
        for i in range(4, -1, -1):
            if s[i] == '?':
                if i == 0:
                    s[i] = '1' if s[i+1] in '0123' else '0'
                elif i == 1:
                    s[i] = '9' if s[i-1] == '1' else '3'
                elif i == 3:
                    s[i] = '5'
                else:
                    s[i] = '9'
        return ''.join(s)