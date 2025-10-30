import math

class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        # Numbers with exactly 2 proper divisors are exactly the squares of primes.
        # So count special numbers = count of primes p where p^2 in [l, r].
        # Answer = total numbers in [l, r] minus that count.

        # Compute integer bounds for p: p^2 >= l  => p >= ceil(sqrt(l))
        #                          and p^2 <= r  => p <= floor(sqrt(r))
        low = math.isqrt(l)
        if low * low < l:
            low += 1
        high = math.isqrt(r)
        if low > high:
            # No squares of primes fit in range
            return r - l + 1

        # Sieve of Eratosthenes up to high
        n = high
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False

        # Count primes in [low, high]
        special_count = sum(1 for p in range(low, high + 1) if is_prime[p])

        # Total numbers minus special ones
        return (r - l + 1) - special_count