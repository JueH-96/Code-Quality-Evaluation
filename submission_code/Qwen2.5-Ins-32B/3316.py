from typing import List
from itertools import combinations

class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        nums.sort()
        n = len(nums)
        result = 0
        
        def min_diff(subseq):
            return min(abs(subseq[i] - subseq[j]) for i in range(len(subseq)) for j in range(i + 1, len(subseq)))
        
        for subseq in combinations(nums, k):
            result = (result + min_diff(subseq)) % MOD
        
        return result