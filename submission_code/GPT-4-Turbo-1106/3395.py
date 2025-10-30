class Solution:
    def minAnagramLength(self, s: str) -> int:
        from collections import Counter
        
        # Count the frequency of each character in the string
        freq = Counter(s)
        
        # The length of t will be the greatest common divisor of all character counts
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        
        # Find the GCD of all character counts
        min_length = 0
        for count in freq.values():
            min_length = gcd(min_length, count)
            if min_length == 1:  # Early stopping if GCD becomes 1
                break
        
        return min_length