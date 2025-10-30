from collections import Counter

class Solution:
    def maxDifference(self, s: str) -> int:
        counts = Counter(s)
        evens = []
        odds = []
        for freq in counts.values():
            if freq % 2 == 0:
                evens.append(freq)
            else:
                odds.append(freq)
        max_odd = max(odds)
        min_even = min(evens)
        return max_odd - min_even