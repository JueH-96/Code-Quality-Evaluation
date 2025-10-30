class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        n = len(s)
        if n == 0:
            return s
        shift = k % n
        res = []
        for i in range(n):
            idx = (i + shift) % n
            res.append(s[idx])
        return "".join(res)