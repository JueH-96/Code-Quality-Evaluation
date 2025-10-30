from typing import List
import math

class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        # Step 1: Sieve of Eratosthenes to find all primes up to 100
        is_prime = [True] * 101
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.sqrt(100)) + 1):
            if is_prime[i]:
                for multiple in range(i*i, 101, i):
                    is_prime[multiple] = False
        primes = set(i for i, prime in enumerate(is_prime) if prime)
        
        # Step 2: Iterate through nums and track min and max indices of primes
        min_index = None
        max_index = None
        for index, num in enumerate(nums):
            if num in primes:
                if min_index is None:
                    min_index = index
                if max_index is None or index > max_index:
                    max_index = index
        
        # Step 3: Calculate and return the maximum distance
        if min_index == max_index:
            return 0
        else:
            return max_index - min_index