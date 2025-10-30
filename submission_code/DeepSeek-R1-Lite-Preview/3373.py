from typing import List

class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        # Precompute primes up to 100
        primes = set([
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
            31, 37, 41, 43, 47, 53, 59, 61, 67,
            71, 73, 79, 83, 89, 97
        ])
        
        prime_indices = []
        
        for index, value in enumerate(nums):
            if value in primes:
                prime_indices.append(index)
        
        if len(prime_indices) <= 1:
            return 0
        else:
            return prime_indices[-1] - prime_indices[0]