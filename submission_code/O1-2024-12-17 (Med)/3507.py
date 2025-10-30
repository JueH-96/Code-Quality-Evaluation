class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        # A "special" number has exactly 3 divisors (1, p, and p^2), so it is the square of a prime.
        # We will count how many prime squares lie in [l, r] and subtract that from the total numbers in [l, r].
        
        import math

        # Upper bound for primes we need to check
        # If p^2 <= r, then p <= sqrt(r)
        limit = int(math.isqrt(r))
        
        # Sieve of Eratosthenes to find primes up to 'limit'
        sieve = [True]*(limit+1)
        sieve[0], sieve[1] = False, False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if sieve[i]:
                for j in range(i*i, limit+1, i):
                    sieve[j] = False
        
        # For each prime p, check if p^2 is in [l, r]
        special_count = 0
        for p in range(2, limit+1):
            if sieve[p]:
                sq = p*p
                if sq > r:
                    break
                if sq >= l:
                    special_count += 1
        
        # Total numbers in [l, r] minus the count of prime squares in that range
        return (r - l + 1) - special_count