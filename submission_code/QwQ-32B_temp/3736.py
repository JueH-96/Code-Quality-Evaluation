from collections import Counter

class Solution:
    def findValidPair(self, s: str) -> str:
        counts = Counter(s)
        for i in range(len(s) - 1):
            d1 = s[i]
            d2 = s[i + 1]
            if d1 != d2 and counts[d1] == int(d1) and counts[d2] == int(d2):
                return d1 + d2
        return ""