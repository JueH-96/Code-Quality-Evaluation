class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        from collections import Counter
        return Counter(s1) == Counter(s2)