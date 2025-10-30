from typing import List

class Solution:
    def minAnagramLength(self, s: str) -> int:
        def get_divisors(n: int) -> List[int]:
            divisors = set()
            for i in range(1, int(n**0.5) + 1):
                if n % i == 0:
                    divisors.add(i)
                    divisors.add(n // i)
            return sorted(divisors)
        
        n = len(s)
        divisors = get_divisors(n)
        
        for k in divisors:
            # Split s into segments of length k
            segments = [s[i:i+k] for i in range(0, n, k)]
            # Sort each segment and check if all are equal
            sorted_segments = [''.join(sorted段)) for 段 in segments]
            if all(x == sorted_segments[0] for x in sorted_segments):
                return k
        return n