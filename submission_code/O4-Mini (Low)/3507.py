import math

class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        # A "special" number has exactly 2 proper divisors,
        # i.e. exactly 3 total divisors.  Those are exactly squares of primes.
        # So we need to count how many prime squares lie in [l, r].
        # Then non-special count = total numbers in range - that count.

        # Compute integer bounds for p such that p^2 in [l, r]
        lo = math.isqrt(l)
        if lo * lo < l:
            lo += 1
        hi = math.isqrt(r)
        if lo > hi:
            # no prime square in the interval
            return r - l + 1

        # Sieve primes up to hi
        n = hi
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False

        # Count primes in [lo, hi]
        cnt_special = sum(1 for p in range(lo, hi+1) if is_prime[p])

        return (r - l + 1) - cnt_special