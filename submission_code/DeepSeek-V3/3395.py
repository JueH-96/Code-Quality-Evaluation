from math import gcd
from functools import reduce

class Solution:
    def minAnagramLength(self, s: str) -> int:
        from collections import Counter
        cnt = Counter(s)
        # Calculate the greatest common divisor (gcd) of all character counts
        def find_gcd_of_list(numbers):
            return reduce(gcd, numbers)
        counts = list(cnt.values())
        g = find_gcd_of_list(counts)
        # The minimum possible length of t is the sum of counts divided by the gcd
        return sum(counts) // g