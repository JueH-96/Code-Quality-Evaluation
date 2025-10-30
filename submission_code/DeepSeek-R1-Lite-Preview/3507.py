import math
import bisect

class Solution:
    primes = []
    
    def __init__(self):
        if not Solution.primes:
            Solution.primes = self.sieve(31623)
            
    def sieve(self, n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for multiple in range(i*i, n+1, i):
                    is_prime[multiple] = False
        primes = [i for i, prime in enumerate(is_prime) if prime]
        return primes
    
    def nonSpecialCount(self, l: int, r: int) -> int:
        A = math.ceil(math.sqrt(l))
        B = math.floor(math.sqrt(r))
        count_primes = bisect.bisect_right(Solution.primes, B) - bisect.bisect_left(Solution.primes, A)
        non_special = (r - l + 1) - count_primes
        return non_special