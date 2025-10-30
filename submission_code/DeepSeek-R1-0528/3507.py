import math

class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        total = r - l + 1
        max_prime_bound = math.isqrt(r)
        if max_prime_bound < 2:
            return total
        is_prime = [True] * (max_prime_bound + 1)
        is_prime[0] = False
        is_prime[1] = False
        for i in range(2, math.isqrt(max_prime_bound) + 1):
            if is_prime[i]:
                for j in range(i*i, max_prime_bound + 1, i):
                    is_prime[j] = False
        primes = [i for i, prime in enumerate(is_prime) if prime]
        count_special = 0
        for p in primes:
            sq = p * p
            if sq < l:
                continue
            if sq > r:
                break
            count_special += 1
        return total - count_special