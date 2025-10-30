from typing import List

class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        # Precompute primes up to 100 (all numbers in nums are <= 100).
        max_val = 100
        is_prime = [True] * (max_val + 1)
        is_prime[0], is_prime[1] = False, False
        for i in range(2, int(max_val**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, max_val+1, i):
                    is_prime[j] = False
                    
        # Collect indices of prime numbers in nums.
        prime_indices = []
        for idx, num in enumerate(nums):
            if is_prime[num]:
                prime_indices.append(idx)
                
        # The maximum difference is the difference between the last and first prime seen.
        return prime_indices[-1] - prime_indices[0]