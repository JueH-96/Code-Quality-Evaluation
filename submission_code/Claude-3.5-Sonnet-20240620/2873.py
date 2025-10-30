class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        def sieve_of_eratosthenes(n):
            primes = [True] * (n + 1)
            primes[0] = primes[1] = False
            for i in range(2, int(n**0.5) + 1):
                if primes[i]:
                    for j in range(i*i, n + 1, i):
                        primes[j] = False
            return primes

        if n < 4:
            return []

        is_prime = sieve_of_eratosthenes(n)
        result = []

        for x in range(2, n // 2 + 1):
            y = n - x
            if is_prime[x] and is_prime[y]:
                result.append([x, y])

        return result