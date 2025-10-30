from typing import List

class Solution:
    primes = None

    @classmethod
    def get_primes(cls):
        if cls.primes is not None:
            return cls.primes
        n = 100
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        cls.primes = {x for x in range(n + 1) if is_prime[x]}
        return cls.primes

    def maximumPrimeDifference(self, nums: List[int]) -> int:
        primes = Solution.get_primes()
        prime_indices = []
        for idx, num in enumerate(nums):
            if num in primes:
                prime_indices.append(idx)
        return prime_indices[-1] - prime_indices[0]