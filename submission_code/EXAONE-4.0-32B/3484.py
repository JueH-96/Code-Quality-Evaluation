class Solution:
    def getSmallestString(self, s: str) -> str:
        n = len(s)
        best = s
        for i in range(n - 1):
            a, b = s[i], s[i + 1]
            if (int(a) % 2) == (int(b) % 2):
                candidate = s[:i] + b + a + s[i+2:]
                if candidate < best:
                    best = candidate
        return best