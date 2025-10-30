class Solution:
    def findLatestTime(self, s: str) -> str:
        s = list(s)
        if s[0] == '?' and s[1] == '?':
            s[0] = '1'
            s[1] = '1'
        elif s[0] == '?':
            if int(s[1]) <= 1:
                s[0] = '1'
            else:
                s[0] = '0'
        elif s[1] == '?':
            if int(s[0]) == 1:
                s[1] = '1'
            else:
                s[1] = '9'
        
        if s[3] == '?':
            s[3] = '5'
        if s[4] == '?':
            s[4] = '9'
        
        return "".join(s)