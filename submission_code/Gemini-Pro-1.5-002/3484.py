class Solution:
    def getSmallestString(self, s: str) -> str:
        n = len(s)
        s = list(s)
        for i in range(n - 1):
            min_idx = i
            for j in range(i + 1, n):
                if int(s[i]) % 2 == int(s[j]) % 2 and s[j] < s[min_idx]:
                    min_idx = j
            if min_idx != i:
                s[i], s[min_idx] = s[min_idx], s[i]
                break
        return "".join(s)