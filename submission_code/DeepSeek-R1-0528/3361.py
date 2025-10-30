class Solution:
    def findLatestTime(self, s: str) -> str:
        res = list(s)
        if res[0] == '?' and res[1] == '?':
            res[0] = '1'
            res[1] = '1'
        elif res[0] == '?':
            if res[1] in '01':
                res[0] = '1'
            else:
                res[0] = '0'
        elif res[1] == '?':
            if res[0] == '1':
                res[1] = '1'
            else:
                res[1] = '9'
        if res[3] == '?' and res[4] == '?':
            res[3] = '5'
            res[4] = '9'
        elif res[3] == '?':
            res[3] = '5'
        elif res[4] == '?':
            res[4] = '9'
        return ''.join(res)