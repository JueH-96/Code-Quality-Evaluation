import collections
from typing import List

class Solution:
    def isGood(self, nums: List[int]) -> bool:
        """
        Determines if an array is a "good" array.
        A good array is a permutation of base[n] = [1, 2, ..., n-1, n, n].
        """
        # The candidate for n is the maximum value in the array, as this is the
        # largest value in base[n]. The constraints ensure nums is not empty.
        n = max(nums)
        
        # A good array must be a permutation of base[n], which has a length of n + 1.
        if len(nums) != n + 1:
            return False
        
        # If the length and max value are consistent, we check the frequencies of elements.
        # We can use a frequency counter for an efficient O(L) solution, where L is len(nums).
        counts = collections.Counter(nums)
        
        # In base[n], the number n must appear exactly twice.
        if counts[n] != 2:
            return False
            
        # In base[n], every number from 1 to n-1 must appear exactly once.
        for i in range(1, n):
            if counts[i] != 1:
                return False
                
        # If all conditions are met, the array is a permutation of base[n].
        return True