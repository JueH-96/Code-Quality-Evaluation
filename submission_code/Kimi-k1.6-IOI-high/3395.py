import math

class Solution:
    def minAnagramLength(self, s: str) -> int:
        n = len(s)
        if n == 0:
            return 0
        
        # Generate all divisors of n and sort them
        divisors = set()
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                divisors.add(i)
                divisors.add(n // i)
        divisors = sorted(divisors)
        
        # Check each divisor starting from the smallest
        for d in divisors:
            k = n // d
            # Compute frequency of the first chunk
            first_freq = [0] * 26
            for c in s[:d]:
                first_freq[ord(c) - ord('a')] += 1
            
            valid = True
            for i in range(1, k):
                start = i * d
                end = start + d
                current_freq = [0] * 26
                for c in s[start:end]:
                    current_freq[ord(c) - ord('a')] += 1
                if current_freq != first_freq:
                    valid = False
                    break
            if valid:
                return d
        
        return n  # This line is theoretically unreachable due to problem constraints