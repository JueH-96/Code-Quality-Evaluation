import math
from bisect import bisect_left, bisect_right

class Solution:
    primes = None
    
    def nonSpecialCount(self, l: int, r: int) -> int:
        # Precompute primes once using Sieve of Eratosthenes
        if Solution.primes is None:
            max_limit = 31623
            sieve = [True] * (max_limit + 1)
            sieve[0] = sieve[1] = False
            for i in range(2, int(math.isqrt(max_limit)) + 1):
                if sieve[i]:
                    sieve[i*i : max_limit + 1 : i] = [False] * len(sieve[i*i : max_limit + 1 : i])
            primes_list = [i for i, is_prime in enumerate(sieve) if is_prime]
            Solution.primes = primes_list
        
        # Calculate lower and upper bounds
        lower = math.isqrt(l)
        if lower * lower < l:
            lower += 1
        upper = math.isqrt(r)
        
        if lower > upper:
            return r - l + 1
        
        # Find the number of primes in [lower, upper]
        left = bisect_left(Solution.primes, lower)
        right = bisect_right(Solution.primes, upper)
        count_special = right - left
        
        total = r - l + 1
        return total - count_special