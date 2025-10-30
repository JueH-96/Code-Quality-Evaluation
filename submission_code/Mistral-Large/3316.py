from typing import List
from itertools import combinations

class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        total_power = 0

        # Generate all subsequences of length k
        for subsequence in combinations(nums, k):
            # Calculate the power of the current subsequence
            min_diff = float('inf')
            subsequence = sorted(subsequence)
            for i in range(1, k):
                min_diff = min(min_diff, subsequence[i] - subsequence[i - 1])
            total_power += min_diff
            total_power %= MOD

        return total_power