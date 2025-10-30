from collections import Counter

class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        count = Counter(s)
        return len(count)