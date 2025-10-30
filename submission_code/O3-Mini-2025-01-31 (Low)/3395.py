import math
from collections import Counter

class Solution:
    def minAnagramLength(self, s: str) -> int:
        # Count frequency of each letter
        freq = Counter(s)
        
        # Compute gcd of all frequencies
        common_gcd = 0
        for count in freq.values():
            common_gcd = count if common_gcd == 0 else math.gcd(common_gcd, count)
        
        # The minimum possible length of t
        return len(s) // common_gcd