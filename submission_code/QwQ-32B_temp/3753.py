from collections import Counter

class Solution:
    def maxDifference(self, s: str) -> int:
        counts = Counter(s)
        evens = []
        odds = []
        for count in counts.values():
            if count % 2 == 0:
                evens.append(count)
            else:
                odds.append(count)
        max_odd = max(odds)
        min_even = min(evens)
        return max_odd - min_even