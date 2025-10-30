from itertools import combinations
from typing import List

class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        total_power = 0

        # Generate all subsequences of length k
        for subsequence in combinations(nums, k):
            # Calculate the power of the subsequence
            min_diff = float('inf')
            for i in range(k):
                for j in range(i + 1, k):
                    min_diff = min(min_diff, abs(subsequence[i] - subsequence[j]))
            total_power = (total_power + min_diff) % MOD

        return total_power