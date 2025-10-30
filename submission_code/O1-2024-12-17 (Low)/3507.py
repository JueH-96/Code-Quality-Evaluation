class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        import math

        # A "special" number has exactly 2 proper divisors.
        # Equivalently, a number x is "special" if x = p^2 for some prime p.
        # Reasoning: p^2 has divisors 1, p, p^2.  =>  proper divisors: 1, p.

        # We want the count of numbers in [l, r] that are NOT special.

        # Approach:
        # 1. Generate all prime numbers up to sqrt(r).
        # 2. Count how many squares of these primes lie within [l, r].
        # 3. The number of special numbers in [l, r] is that count.
        # 4. The final result = (r - l + 1) - (count of prime squares in [l, r]).

        def sieve_of_eratosthenes(n):
            """ Generate primes up to n (inclusive) using the sieve of Eratosthenes. """
            sieve = [True] * (n + 1)
            sieve[0] = sieve[1] = False
            for i in range(2, int(n**0.5) + 1):
                if sieve[i]:
                    for j in range(i*i, n + 1, i):
                        sieve[j] = False
            return [i for i in range(2, n + 1) if sieve[i]]

        # Maximum possible value of r is 10^9, so we need primes up to sqrt(r).
        limit = int(math.isqrt(r))
        primes = sieve_of_eratosthenes(limit)

        count_special = 0
        for p in primes:
            square = p * p
            if square > r:
                break
            if square >= l:
                count_special += 1

        # Total numbers in [l, r] = r - l + 1
        # Subtract the count of special numbers to get count of non-special numbers.
        return (r - l + 1) - count_special