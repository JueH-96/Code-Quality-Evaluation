import math

class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        if l > r:
            return 0
        
        p_max = math.isqrt(r)
        sieve = [True] * (p_max + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, math.isqrt(p_max) + 1):
            if sieve[i]:
                sieve[i*i : p_max+1 : i] = [False] * len(sieve[i*i : p_max+1 : i])
        
        primes = [i for i, is_prime in enumerate(sieve) if is_prime]
        
        count_special = 0
        for p in primes:
            square = p * p
            if square >= l and square <= r:
                count_special += 1
        
        return (r - l + 1) - count_special