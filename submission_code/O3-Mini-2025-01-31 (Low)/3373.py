from typing import List

class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        # Function to check for primes using a precomputed sieve up to 100
        max_val = 100
        sieve = [True] * (max_val + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(max_val**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, max_val + 1, i):
                    sieve[j] = False
        
        # Find indices for prime numbers in the list
        prime_indices = [i for i, value in enumerate(nums) if sieve[value]]
        
        # The maximum distance is between the smallest and the largest index with a prime number.
        # As there's guaranteed to be at least one prime, we can take the first and last.
        return prime_indices[-1] - prime_indices[0]