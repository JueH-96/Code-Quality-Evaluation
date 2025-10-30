from typing import List

class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}
        primes_indices = []
        
        for idx, num in enumerate(nums):
            if num in primes:
                primes_indices.append(idx)
        
        if len(primes_indices) <= 1:
            return 0
        else:
            return primes_indices[-1] - primes_indices[0]