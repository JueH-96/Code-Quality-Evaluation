from collections import Counter

class Solution:
    def minimumLength(self, s: str) -> int:
        counts = Counter(s)
        total = 0
        for c in counts:
            cnt = counts[c]
            if cnt % 2 == 1:
                total += 1
            else:
                total += 2
        return total