from math import gcd
from functools import reduce

class Solution:
    def minAnagramLength(self, s: str) -> int:
        # Count the frequency of each character in the string
        freq = {}
        for char in s:
            if char in freq:
                freq[char] += 1
            else:
                freq[char] = 1
        
        # Find the greatest common divisor (GCD) of all frequencies
        frequencies = list(freq.values())
        overall_gcd = reduce(gcd, frequencies)
        
        # The minimum possible length of t is the sum of frequencies divided by the GCD
        return sum(frequencies) // overall_gcd