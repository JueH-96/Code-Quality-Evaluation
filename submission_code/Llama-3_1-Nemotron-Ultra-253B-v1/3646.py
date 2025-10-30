from typing import List
from collections import defaultdict

MOD = 10**9 + 7

class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        dp = defaultdict(lambda: (0, 0))  # (sum, count)
        total = 0
        for x in nums:
            sum_prev = 0
            count_prev = 0
            for y in [x - 1, x + 1]:
                if y in dp:
                    s, c = dp[y]
                    sum_prev = (sum_prev + s + c * x) % MOD
                    count_prev = (count_prev + c) % MOD
            new_sum = (sum_prev + x) % MOD
            new_count = (count_prev + 1) % MOD
            total = (total + new_sum) % MOD
            # Update dp[x]
            s_old, c_old = dp[x]
            dp[x] = ((s_old + new_sum) % MOD, (c_old + new_count) % MOD)
        return total % MOD