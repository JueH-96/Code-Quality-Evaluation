from collections import Counter

class Solution:
    def maxDifference(self, s: str) -> int:
        counts = Counter(s)
        even = []
        odd = []
        for char, freq in counts.items():
            if freq % 2 == 0:
                even.append(freq)
            else:
                odd.append(freq)
        max_odd = max(odd)
        min_even = min(even)
        return max_odd - min_even