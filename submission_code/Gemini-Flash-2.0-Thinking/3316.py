from typing import List
from itertools import combinations

class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        total_power = 0

        for subsequence_tuple in combinations(nums, k):
            subsequence = list(subsequence_tuple)
            min_diff = float('inf')
            if len(subsequence) >= 2:
                for i in range(len(subsequence)):
                    for j in range(i + 1, len(subsequence)):
                        diff = abs(subsequence[i] - subsequence[j])
                        min_diff = min(min_diff, diff)
                total_power = (total_power + min_diff) % MOD

        return total_power