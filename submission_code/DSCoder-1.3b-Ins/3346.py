class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        res = []
        for i in range(len(s)):
            if s[i] == '1':
                res.append('a')
                k -= 1
            elif s[i] == '0':
                res.append('b')
                k -= 1
            else:
                if k >= 2:
                    res.append('a')
                    k -= 1
                else:
                    res.append('b')
                    k -= 1
        if k > 0:
            res.append('a' * k)
        return ''.join(res)