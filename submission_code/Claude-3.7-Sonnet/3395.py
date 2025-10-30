class Solution:
    def minAnagramLength(self, s: str) -> int:
        from math import gcd
        from functools import reduce
        from collections import Counter
        
        # Count the frequency of each character in s
        freq = Counter(s)
        
        # Find the gcd of all frequencies
        gcd_val = reduce(gcd, freq.values())
        
        # The minimum length of t is len(s) / gcd_val
        return len(s) // gcd_val