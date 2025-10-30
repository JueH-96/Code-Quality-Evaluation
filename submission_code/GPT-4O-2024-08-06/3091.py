from typing import List
from collections import Counter

MOD = 10**9 + 7

class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        # Calculate the number of ways to form subsets with sum <= x
        def count_subsets_with_sum_at_most(x):
            dp = [0] * (x + 1)
            dp[0] = 1  # There's one way to make sum 0: the empty set

            for num in nums:
                for j in range(x, num - 1, -1):
                    dp[j] = (dp[j] + dp[j - num]) % MOD

            return sum(dp) % MOD

        # Count subsets with sum <= r and subtract those with sum < l
        count_r = count_subsets_with_sum_at_most(r)
        count_l_minus_1 = count_subsets_with_sum_at_most(l - 1)
        
        return (count_r - count_l_minus_1 + MOD) % MOD