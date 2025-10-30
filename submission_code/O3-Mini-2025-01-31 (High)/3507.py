import math

class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        # Total numbers in the range [l, r]
        total = r - l + 1
        
        # Special numbers are the ones with exactly 2 proper divisors.
        # Such numbers have exactly 3 divisors (including the number itself).
        # It is known that numbers with exactly 3 divisors are perfect squares of primes.
        # So we need to count those x = p^2 that fall within [l, r].
        
        # The largest possible prime p we need is when p^2 <= r, i.e., p <= sqrt(r)
        max_p = math.isqrt(r)
        
        # Sieve to generate primes up to max_p
        sieve = [True] * (max_p + 1)
        sieve[0] = sieve[1] = False  # 0 and 1 are not primes
        for i in range(2, int(max_p**0.5) + 1):
            if sieve[i]:
                for j in range(i * i, max_p + 1, i):
                    sieve[j] = False

        # Count the special numbers (i.e., squares of a prime within [l, r])
        special_count = 0
        for p in range(2, max_p + 1):
            if sieve[p]:
                square = p * p
                if l <= square <= r:
                    special_count += 1

        # Return the count of non-special numbers
        return total - special_count