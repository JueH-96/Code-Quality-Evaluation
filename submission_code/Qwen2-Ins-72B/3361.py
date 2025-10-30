class Solution:
    def findLatestTime(self, s: str) -> str:
        res = list(s)
        if res[0] == '?':
            if res[1] == '?':
                if int(s[3:5]) < 60:
                    res[0], res[1] = '2', '3'
                else:
                    res[0], res[1] = '1', '9'
            else:
                if res[1] <= '3':
                    res[0] = '2'
                else:
                    res[0] = '1'
        if res[1] == '?':
            if res[0] == '2':
                res[1] = '3'
            else:
                res[1] = '9'
        if res[3] == '?':
            res[3] = '5'
        if res[4] == '?':
            res[4] = '9'
        return ''.join(res)