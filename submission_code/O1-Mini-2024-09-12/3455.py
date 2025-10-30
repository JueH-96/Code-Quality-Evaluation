class Solution:
    def minimumLength(self, s: str) -> int:
        from collections import Counter

        counts = Counter(s)
        total_deletions = 0
        for cnt in counts.values():
            deletions = (cnt - 1) // 2
            total_deletions += deletions * 2
        return len(s) - total_deletions