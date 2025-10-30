import math

class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        max_p = math.isqrt(r)
        primes = []
        if max_p >= 2:
            sieve = [True] * (max_p + 1)
            sieve[0] = sieve[1] = False
            for i in range(2, int(math.isqrt(max_p)) + 1):
                if sieve[i]:
                    sieve[i*i : max_p+1 : i] = [False] * len(sieve[i*i : max_p+1 : i])
            primes = [p for p in range(len(sieve)) if sieve[p]]
        special_count = 0
        for p in primes:
            if p * p >= l:
                special_count += 1
        total = r - l + 1
        return total - special_count