class Solution:
    def findLatestTime(self, s: str) -> str:
        s = list(s)
        for i in range(len(s)):
            if s[i] == '?':
                if i == 0:
                    if s[1] <= '3' or s[1] == '?':
                        s[i] = '2'
                    else:
                        s[i] = '1'
                elif i == 1:
                    if s[0] == '1' or s[0] == '0':
                        s[i] = '9'
                    else:
                        s[i] = '3'
                elif i == 3:
                    s[i] = '5'
                elif i == 4:
                    s[i] = '9'
        return ''.join(s)