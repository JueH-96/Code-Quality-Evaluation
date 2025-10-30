from typing import List

class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        primes = set([
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
            43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
        ])
        first = last = None
        for i, num in enumerate(nums):
            if num in primes:
                if first is None:
                    first = i
                last = i
        return last - first