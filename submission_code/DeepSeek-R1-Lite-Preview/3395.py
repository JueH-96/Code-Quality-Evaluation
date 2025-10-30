import math
from collections import Counter

class Solution:
    def minAnagramLength(self, s: str) -> int:
        # Step 1: Calculate the frequency counts of each character in s
        count = Counter(s)
        
        # Step 2: Compute the GCD of all frequency counts
        from math import gcd
        from functools import reduce
        gcd_counts = reduce(gcd, count.values())
        
        # Step 3: Calculate the number of parts k
        k = gcd_counts
        
        # Step 4: Calculate the minimal length of t
        m = len(s) // k
        return m