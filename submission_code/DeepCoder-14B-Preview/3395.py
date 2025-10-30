import math
from collections import Counter

class Solution:
    def minAnagramLength(self, s: str) -> int:
        if not s:
            return 0
        n = len(s)
        freq = Counter(s)
        
        # Generate all divisors of n
        divisors = set()
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                divisors.add(i)
                divisors.add(n // i)
        # Sort the divisors in increasing order
        sorted_divisors = sorted(divisors)
        
        for k in sorted_divisors:
            m = n // k
            # Check all frequencies are divisible by m
            all_ok = True
            for c in freq:
                if freq[c] % m != 0:
                    all_ok = False
                    break
            if all_ok:
                return k
        # If no k found (which shouldn't happen as k=n is a divisor)
        return n