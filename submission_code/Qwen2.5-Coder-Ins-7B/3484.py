class Solution:
    def getSmallestString(self, s: str) -> str:
        s = list(s)
        n = len(s)
        for i in range(n):
            if i % 2 == 0:
                min_idx = i
                for j in range(i + 2, n, 2):
                    if int(s[j]) < int(s[min_idx]):
                        min_idx = j
                if min_idx != i:
                    s[i], s[min_idx] = s[min_idx], s[i]
            else:
                min_idx = i
                for j in range(i + 2, n, 2):
                    if int(s[j]) < int(s[min_idx]):
                        min_idx = j
                if min_idx != i:
                    s[i], s[min_idx] = s[min_idx], s[i]
        return ''.join(s)