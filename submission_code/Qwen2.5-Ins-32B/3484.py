class Solution:
    def getSmallestString(self, s: str) -> str:
        s = list(s)
        n = len(s)
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if int(s[j]) % 2 == int(s[i]) % 2 and s[j] < s[min_idx]:
                    min_idx = j
            if min_idx != i:
                s[i], s[min_idx] = s[min_idx], s[i]
                break
        return ''.join(s)