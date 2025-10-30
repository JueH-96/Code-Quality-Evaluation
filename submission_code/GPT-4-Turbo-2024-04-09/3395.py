class Solution:
    def minAnagramLength(self, s: str) -> int:
        from collections import Counter
        
        # Count frequency of each character in the string s
        char_count = Counter(s)
        
        # The minimum length of t is determined by the greatest common divisor (GCD) of all character counts
        from math import gcd
        from functools import reduce
        
        # Get all counts from the counter
        counts = list(char_count.values())
        
        # Calculate the GCD of all counts
        overall_gcd = reduce(gcd, counts)
        
        return overall_gcd