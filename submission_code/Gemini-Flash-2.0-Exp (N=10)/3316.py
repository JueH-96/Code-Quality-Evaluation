from typing import List
import itertools

class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        if k > n:
            return 0

        total_power = 0
        for subseq in itertools.combinations(nums, k):
            min_diff = float('inf')
            for i in range(k):
                for j in range(i + 1, k):
                    min_diff = min(min_diff, abs(subseq[i] - subseq[j]))
            total_power = (total_power + min_diff) % MOD
        
        return total_power