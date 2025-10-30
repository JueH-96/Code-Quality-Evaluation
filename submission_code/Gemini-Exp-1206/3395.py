class Solution:
    def minAnagramLength(self, s: str) -> int:
        from collections import Counter
        from math import gcd

        count = Counter(s)
        g = 0
        for c in count.values():
            g = gcd(g, c)
        return len(s) // g