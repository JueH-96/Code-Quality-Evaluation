from typing import List

class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        # Sieve of Eratosthenes to find all primes up to n
        is_prime = [True] * (n + 1)
        if n >= 0:
            is_prime[0] = False
        if n >= 1:
            is_prime[1] = False

        p = 2
        while p * p <= n:
            if is_prime[p]:
                for multiple in range(p * p, n + 1, p):
                    is_prime[multiple] = False
            p += 1

        # Collect pairs (x, y) with x <= y, x + y = n, both prime
        res = []
        # Only need to check x up to n//2 to ensure x <= y
        for x in range(2, n // 2 + 1):
            if is_prime[x]:
                y = n - x
                if y >= x and is_prime[y]:
                    res.append([x, y])

        return res