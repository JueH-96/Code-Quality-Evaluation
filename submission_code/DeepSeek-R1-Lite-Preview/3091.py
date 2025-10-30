from typing import List
from collections import Counter

class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        freq = Counter(nums)
        sum_nums = sum(nums)
        dp = [0] * (sum_nums + 1)
        dp[0] = 1  # Empty subset

        for num, count in freq.items():
            # Iterate from high to low to prevent using the same number more times than allowed
            for s in range(sum_nums, -1, -1):
                for k in range(1, count + 1):
                    if s + k * num > sum_nums:
                        break
                    dp[s + k * num] = (dp[s + k * num] + dp[s]) % MOD

        # Sum the counts of all valid sums in [l, r]
        total = 0
        for s in range(l, r + 1):
            if s <= sum_nums:
                total = (total + dp[s]) % MOD
        return total