from collections import Counter
from math import gcd

class Solution:
    def minAnagramLength(self, s: str) -> int:
        n = len(s)
        if n == 0:
            return 0

        char_counts = Counter(s)
        frequencies = list(char_counts.values())

        if not frequencies:
            return 0

        result_gcd = frequencies[0]
        for i in range(1, len(frequencies)):
            result_gcd = gcd(result_gcd, frequencies[i])

        return n // result_gcd