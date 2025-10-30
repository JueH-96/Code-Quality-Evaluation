class Solution:
    def findLatestTime(self, s: str) -> str:
        s = s.replace('?', '9')
        if int(s[:2]) > 11:
            s = '1' + s[1:]
            if int(s[1]) > 1:
                s = s[0] + '1' + s[2:]
            s = s.replace(';', ':')
        else:
            s = s.replace(';', ':')
        return s