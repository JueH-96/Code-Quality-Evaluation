import math
import bisect

class Solution:
    _primes = None

    def __init__(self):
        if Solution._primes is None:
            n = 31622  # since sqrt(1e9) is about 31622.7766
            sieve = [True] * (n + 1)
            sieve[0], sieve[1] = False, False
            for i in range(2, int(math.isqrt(n)) + 1):
                if sieve[i]:
                    sieve[i*i : n+1 : i] = [False] * len(sieve[i*i : n+1 : i])
            Solution._primes = [i for i, is_prime in enumerate(sieve) if is_prime]

    def nonSpecialCount(self, l: int, r: int) -> int:
        lower_sq = math.isqrt(l)
        if lower_sq * lower_sq < l:
            lower = lower_sq + 1
        else:
            lower = lower_sq
        upper = math.isqrt(r)
        if lower > upper:
            count_special = 0
        else:
            primes = Solution._primes
            left = bisect.bisect_left(primes, lower)
            right = bisect.bisect_right(primes, upper)
            count_special = right - left
        total = r - l + 1
        return total - count_special