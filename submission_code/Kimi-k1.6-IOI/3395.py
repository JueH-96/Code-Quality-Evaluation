import math

class Solution:
    def minAnagramLength(self, s: str) -> int:
        n = len(s)
        if n == 0:
            return 0
        
        # Precompute prefix sums for each character (a-z)
        prefix = [[0] * (n + 1) for _ in range(26)]
        for i in range(n):
            for c in range(26):
                prefix[c][i+1] = prefix[c][i] + (1 if s[i] == chr(ord('a') + c) else 0)
        
        # Generate all divisors of n sorted in increasing order
        divisors = set()
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                divisors.add(i)
                divisors.add(n // i)
        divisors = sorted(divisors)
        
        # Check each divisor in order
        for d in divisors:
            m = n // d
            # Compute frequency tuple for the first chunk
            start = 0
            end = d
            target = tuple(prefix[c][end] - prefix[c][start] for c in range(26))
            all_match = True
            for i in range(1, m):
                start = i * d
                end = start + d
                curr_tuple = tuple(prefix[c][end] - prefix[c][start] for c in range(26))
                if curr_tuple != target:
                    all_match = False
                    break
            if all_match:
                return d
        return n