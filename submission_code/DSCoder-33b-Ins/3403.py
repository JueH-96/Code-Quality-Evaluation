class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        from collections import Counter
        count = Counter(s)
        return max(count.values())