from collections import Counter

class Solution:
    def findValidPair(self, s: str) -> str:
        count = Counter(s)
        n = len(s)
        for i in range(n - 1):
            a, b = s[i], s[i+1]
            if a == b:
                continue
            if count[a] == int(a) and count[b] == int(b):
                return a + b
        return ""