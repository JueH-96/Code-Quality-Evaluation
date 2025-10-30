class Solution:
    def minimumLength(self, s: str) -> int:
        from collections import Counter
        freq = Counter(s)
        removable = 0
        for count in freq.values():
            if count >= 3:
                removable += 2 * (count // 2)
        return len(s) - removable