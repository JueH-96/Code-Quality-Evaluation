from typing import List
from collections import Counter

class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        max_sum = 2 * 10**4
        dp = [[0] * (max_sum + 1) for _ in range(len(nums) + 1)]
        dp[0][0] = 1

        for num in nums:
            for s in range(max_sum, num - 1, -1):
                for i in range(1, len(nums) + 1):
                    dp[i][s] = (dp[i][s] + dp[i-1][s-num]) % MOD

        return (sum(dp[i][s] for i in range(1, len(nums) + 1) for s in range(l, r + 1)) - dp[len(nums)][0]) % MOD