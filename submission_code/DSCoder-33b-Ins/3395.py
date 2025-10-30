class Solution:
    def minAnagramLength(self, s: str) -> int:
        from collections import Counter
        count = Counter(s)
        return len(s) - max(count.values())