from typing import List
from itertools import combinations

class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        nums.sort()
        total_sum = 0
        
        for subseq in combinations(nums, k):
            min_diff = float('inf')
            for i in range(1, k):
                min_diff = min(min_diff, abs(subseq[i] - subseq[i-1]))
            total_sum = (total_sum + min_diff) % MOD
        
        return total_sum