from collections import Counter

class Solution:
    def minimumLength(self, s: str) -> int:
        count = Counter(s)
        total_pairs = sum((v - 1) // 2 for v in count.values())
        return len(s) - 2 * total_pairs