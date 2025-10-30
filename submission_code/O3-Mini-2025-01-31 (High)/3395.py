class Solution:
    def minAnagramLength(self, s: str) -> int:
        from collections import Counter
        from math import gcd
        
        # Count frequency of each character in s.
        freq = Counter(s)
        
        # Compute the gcd of all non-zero character counts.
        current_gcd = 0
        for count in freq.values():
            current_gcd = gcd(current_gcd, count)
        
        # The string s is k anagrams of t.
        # For every character x: count_in_t = freq[x] / k, where k = current_gcd.
        # Thus len(t) = sum(freq[x] / k) = len(s) / k.
        return len(s) // current_gcd