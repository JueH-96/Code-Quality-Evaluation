class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        import math

        def sieve(n):
            is_prime = [True] * (n + 1)
            is_prime[0] = is_prime[1] = False
            for i in range(2, int(math.sqrt(n)) + 1):
                if is_prime[i]:
                    for multiple in range(i*i, n+1, i):
                        is_prime[multiple] = False
            primes = [i for i, prime in enumerate(is_prime) if prime]
            return primes

        sqrt_r = int(math.isqrt(r))
        primes = sieve(sqrt_r)
        count_special = 0
        for p in primes:
            p_sq = p * p
            if l <= p_sq <= r:
                count_special += 1
        total = r - l + 1
        return total - count_special