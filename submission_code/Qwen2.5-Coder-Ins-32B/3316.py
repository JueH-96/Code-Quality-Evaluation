from itertools import combinations
from typing import List

class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        total_sum = 0
        
        for subseq in combinations(nums, k):
            min_diff = float('inf')
            for i in range(len(subseq)):
                for j in range(i + 1, len(subseq)):
                    min_diff = min(min_diff, abs(subseq[i] - subseq[j]))
            total_sum = (total_sum + min_diff) % MOD
        
        return total_sum