import math

class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        max_p = math.isqrt(r)
        primes = self.sieve(max_p)
        count = 0
        for p in primes:
            square = p * p
            if l <= square <= r:
                count += 1
        total = r - l + 1
        return total - count
    
    def sieve(self, n):
        if n < 2:
            return []
        is_prime = [True] * (n + 1)
        is_prime[0], is_prime[1] = False, False
        for i in range(2, math.isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [x for x, val in enumerate(is_prime) if val]