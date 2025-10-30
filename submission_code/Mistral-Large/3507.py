class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        def sieve(n):
            is_prime = [True] * (n + 1)
            is_prime[0] = is_prime[1] = False
            for i in range(2, int(n**0.5) + 1):
                if is_prime[i]:
                    for j in range(i * i, n + 1, i):
                        is_prime[j] = False
            return [i for i in range(n + 1) if is_prime[i]]

        max_sqrt = int(r**0.5)
        primes = sieve(max_sqrt)

        special_numbers = set(p * p for p in primes if p * p >= l)

        non_special_count = 0
        for num in range(l, r + 1):
            if num not in special_numbers:
                non_special_count += 1

        return non_special_count