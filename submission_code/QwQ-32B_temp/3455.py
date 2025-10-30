from collections import Counter

class Solution:
    def minimumLength(self, s: str) -> int:
        counts = Counter(s)
        total_ops = 0
        for c in counts.values():
            total_ops += (c - 1) // 2
        return len(s) - 2 * total_ops