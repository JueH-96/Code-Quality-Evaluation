import math
from typing import List

class Solution:
    def minimumCost(self, nums: List[int], cost: List[int], k: int) -> int:
        n = len(nums)
        # Precompute prefix sums for nums and cost
        prefix_sum_nums = [0]
        cum_sum_num = 0
        for num in nums:
            cum_sum_num += num
            prefix_sum_nums.append(cum_sum_num)
        
        prefix_cost_sum = [0]
        cum_cost_sum = 0
        for c in cost:
            cum_cost_sum += c
            prefix_cost_sum.append(cum_cost_sum)
        
        # Initialize dp table with infinity
        dp = [[float('inf') for _ in range(n + 1)] for _ in range(n + 1)]
        dp[0][0] = 0
        
        # Fill dp table
        for num_sub in range(1, n + 1):
            for length in range(num_sub, n + 1):
                A = prefix_sum_nums[length] + k * num_sub
                min_cost = float('inf')
                for s in range(num_sub - 1, length):
                    cost_add = dp[s][num_sub - 1] + A * (prefix_cost_sum[length] - prefix_cost_sum[s])
                    if cost_add < min_cost:
                        min_cost = cost_add
                dp[length][num_sub] = min_cost
        
        # The answer is the minimum cost over all possible number of subarrays
        return min(dp[n][m] for m in range(1, n + 1))