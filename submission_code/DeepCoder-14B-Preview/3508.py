import math

class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        total = r - l + 1
        max_p = int(math.isqrt(r))
        
        if max_p < 2:
            return total
        
        # Sieve of Eratosthenes to find all primes up to max_p
        sieve = [True] * (max_p + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.isqrt(max_p)) + 1):
            if sieve[i]:
                sieve[i*i : max_p + 1 : i] = [False] * len(sieve[i*i : max_p + 1 : i])
        
        primes = [i for i, is_p in enumerate(sieve) if is_p]
        
        count = 0
        for p in primes:
            p_squared = p * p
            if l <= p_squared <= r:
                count += 1
        
        return total - count