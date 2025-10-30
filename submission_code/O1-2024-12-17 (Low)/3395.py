class Solution:
    def minAnagramLength(self, s: str) -> int:
        from math import gcd
        from collections import Counter
        
        # Count the frequency of each character in s
        freq = Counter(s)
        
        # Compute the GCD of all non-zero frequencies
        current_gcd = 0
        for c in freq:
            current_gcd = gcd(current_gcd, freq[c])
        
        # The minimal length of t is the sum of freq[c] // current_gcd
        return sum(f // current_gcd for f in freq.values())