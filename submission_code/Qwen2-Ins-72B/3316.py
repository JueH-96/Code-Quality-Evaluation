from itertools import combinations
from math import inf
from typing import List

class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        total_power = 0
        
        # Generate all combinations of nums of length k
        for subseq in combinations(nums, k):
            min_diff = inf
            
            # Calculate the minimum absolute difference for the current subsequence
            for i in range(len(subseq) - 1):
                min_diff = min(min_diff, abs(subseq[i] - subseq[i + 1]))
            
            # Add the minimum absolute difference to the total power
            total_power = (total_power + min_diff) % MOD
        
        return total_power