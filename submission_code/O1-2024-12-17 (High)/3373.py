from typing import List

class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        # Precompute all primes up to 100
        prime_set = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
                     31, 37, 41, 43, 47, 53, 59, 61, 67,
                     71, 73, 79, 83, 89, 97}
        
        first_prime_index = None
        last_prime_index = None
        
        for i, val in enumerate(nums):
            if val in prime_set:
                if first_prime_index is None:
                    first_prime_index = i
                last_prime_index = i
        
        return last_prime_index - first_prime_index