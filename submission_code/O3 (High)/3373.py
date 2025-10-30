from typing import List

class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        # Pre-computed set of all primes ≤ 100 (the maximum possible value)
        PRIMES = {
             2,  3,  5,  7, 11, 13, 17, 19, 23, 29,
            31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
            73, 79, 83, 89, 97
        }
        
        first = -1   # index of first prime encountered
        last  = -1   # index of last prime encountered
        
        for i, val in enumerate(nums):
            if val in PRIMES:
                if first == -1:
                    first = i
                last = i          # always update last
        
        # The problem guarantees at least one prime number.
        return last - first