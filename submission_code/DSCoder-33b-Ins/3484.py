class Solution:
    def getSmallestString(self, s: str) -> str:
        s = list(s)
        n = len(s)
        for i in range(n):
            if int(s[i]) % 2 == 0:
                continue
            for j in range(i+1, n):
                if int(s[j]) % 2 == 0:
                    s[i], s[j] = s[j], s[i]
                    return ''.join(s)
        return ''.join(s)