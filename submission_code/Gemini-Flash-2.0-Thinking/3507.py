class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        def sieve(limit):
            primes = []
            is_prime = [True] * (limit + 1)
            for p in range(2, limit + 1):
                if is_prime[p]:
                    primes.append(p)
                    for i in range(p * p, limit + 1, p):
                        is_prime[i] = False
            return primes

        sqrt_r = int(r**0.5)
        primes = sieve(sqrt_r)

        special_count = 0
        for p in primes:
            square_p = p * p
            if l <= square_p <= r:
                special_count += 1

        total_count = r - l + 1
        non_special_count = total_count - special_count
        return non_special_count