class Solution:
    def getSmallestString(self, s: str) -> str:
        n = len(s)
        s = list(s)
        for i in range(n):
            max_j = i
            for j in range(i, n):
                if int(s[j]) % 2 == int(s[i]) % 2 and int(s[j]) < int(s[max_j]):
                    max_j = j
            if max_j != i:
                s[i], s[max_j] = s[max_j], s[i]
        return ''.join(s)