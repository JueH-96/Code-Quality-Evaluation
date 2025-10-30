from collections import Counter

class Solution:
    def minimumLength(self, s: str) -> int:
        counts = Counter(s)
        result = 0
        for count in counts.values():
            if count == 1:
                result += 1
            elif count % 2 == 0:
                result += 2
            else:
                result += 1
        return result