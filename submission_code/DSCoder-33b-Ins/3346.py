class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        n = len(s)
        res = ['a'] * n
        k -= n
        i = n - 1
        while k > 0:
            add = min(25, k)
            res[i] = chr(ord(res[i]) + add)
            k -= add
            i -= 1
        return ''.join(res)