from collections import Counter

class Solution:
    def findValidPair(self, s: str) -> str:
        counts = Counter(s)
        n = len(s)
        for i in range(n - 1):
            d1, d2 = s[i], s[i+1]
            # digits must be different
            if d1 == d2:
                continue
            # each digit's total count must equal its numeric value
            if counts[d1] == int(d1) and counts[d2] == int(d2):
                return d1 + d2
        return ""