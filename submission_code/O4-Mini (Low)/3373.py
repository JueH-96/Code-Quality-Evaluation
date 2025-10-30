from typing import List

class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        # Precomputed set of primes up to 100
        primes = {
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
            31, 37, 41, 43, 47, 53, 59, 61, 67,
            71, 73, 79, 83, 89, 97
        }
        
        first_idx = None
        last_idx = None
        
        for i, val in enumerate(nums):
            if val in primes:
                if first_idx is None:
                    first_idx = i
                last_idx = i
        
        # At least one prime is guaranteed in nums
        return last_idx - first_idx