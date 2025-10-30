import math

class Solution:
    def minAnagramLength(self, s: str) -> int:
        n = len(s)
        divisors = set()
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                divisors.add(i)
                divisors.add(n // i)
        sorted_divisors = sorted(divisors)
        for k in sorted_divisors:
            ref = sorted(s[:k])
            m = n // k
            valid = True
            for i in range(1, m):
                current = s[i*k : (i+1)*k]
                if sorted(current) != ref:
                    valid = False
                    break
            if valid:
                return k
        return n