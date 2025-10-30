class Solution:
    def findLatestTime(self, s: str) -> str:
        s = list(s)
        # Process hours
        if s[0] == '?':
            if s[1] == '?':
                s[0] = '1'
                s[1] = '1'
            elif '0' <= s[1] <= '1':
                s[0] = '1'
            else:
                s[0] = '0'
        else:
            if s[0] == '1':
                if s[1] != '?' and s[1] > '1':
                    s[0] = '0'
            elif s[0] == '0':
                pass
        if s[1] == '?':
            if s[0] == '1':
                s[1] = '1'
            else:
                s[1] = '9'
        # Process minutes
        if s[3] == '?':
            s[3] = '5'
        if s[4] == '?':
            s[4] = '9'
        return ''.join(s)