import math

class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        """
        A number is *special* iff it has exactly two proper divisors.
        That happens exactly for squares of primes (p², p prime).
        So we have to subtract from the interval length the amount of
        prime squares that fall into [l, r].
        """

        # interval of possible primes whose square is inside [l, r]
        lo = math.isqrt(l)
        if lo * lo < l:                 # ceiling of √l
            lo += 1
        hi = math.isqrt(r)              # floor of √r

        # no square can lie in the range
        if lo > hi:
            return r - l + 1

        # sieve primes up to hi   (hi ≤ √1e9 ≈ 31623, very small)
        limit = hi
        sieve = [True] * (limit + 1)
        if limit >= 0:
            sieve[0] = False
        if limit >= 1:
            sieve[1] = False

        root = int(math.isqrt(limit))
        for p in range(2, root + 1):
            if sieve[p]:
                for multiple in range(p * p, limit + 1, p):
                    sieve[multiple] = False

        # how many primes are in [lo, hi]?
        prime_cnt = sum(1 for x in range(lo, hi + 1) if sieve[x])

        # every such prime contributes one special number (its square)
        total_numbers = r - l + 1
        return total_numbers - prime_cnt