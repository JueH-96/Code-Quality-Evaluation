class Solution:
    def minAnagramLength(self, s: str) -> int:
        from collections import Counter
        char_count = Counter(s)
        return max(list(char_count.values()))