from itertools import combinations
from typing import List

class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        def power(subseq):
            if len(subseq) < 2:
                return 0
            min_diff = min(abs(subseq[i] - subseq[j]) for i in range(len(subseq)) for j in range(i + 1, len(subseq)))
            return min_diff
        
        total_power = 0
        for subseq in combinations(nums, k):
            total_power = (total_power + power(subseq)) % MOD
        
        return total_power