class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        n = len(s)
        dist = 0
        res = list(s)
        for i in range(n):
            diff = ord(s[i]) - ord('a')
            if k >= diff:
                k -= diff
                res[i] = 'a'
            else:
                res[i] = chr(ord(s[i]) - k)
                k = 0
                
        if k > 0:
            for i in range(n - 1, -1, -1):
                add = min(25, k)
                res[i] = chr(ord(res[i]) + add)
                k -= add
                if k == 0:
                    break
        return "".join(res)