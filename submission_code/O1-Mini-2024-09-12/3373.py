from typing import List

class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        # Precompute primes up to 100
        primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
                  31, 37, 41, 43, 47, 53, 59, 61, 67,
                  71, 73, 79, 83, 89, 97}
        
        first = -1
        last = -1
        
        for i, num in enumerate(nums):
            if num in primes:
                first = i
                break
        
        for i in range(len(nums)-1, -1, -1):
            if nums[i] in primes:
                last = i
                break
        
        return last - first if first != -1 and last != -1 else 0