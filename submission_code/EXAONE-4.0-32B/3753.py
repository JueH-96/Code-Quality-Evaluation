from collections import Counter

class Solution:
    def maxDifference(self, s: str) -> int:
        freq = Counter(s)
        evens = []
        odds = []
        for count in freq.values():
            if count % 2 == 0:
                evens.append(count)
            else:
                odds.append(count)
        return max(odds) - min(evens)