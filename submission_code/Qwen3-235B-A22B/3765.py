from typing import List

class Solution:
    def minimumCost(self, nums: List[int], cost: List[int], k: int) -> int:
        n = len(nums)
        # Precompute prefix sums for nums and cost
        pre_num = [0] * (n + 1)
        pre_cost = [0] * (n + 1)
        for i in range(1, n + 1):
            pre_num[i] = pre_num[i-1] + nums[i-1]
            pre_cost[i] = pre_cost[i-1] + cost[i-1]
        
        # Initialize DP table with infinity
        dp = [[float('inf')] * (n + 1) for _ in range(n + 1)]
        dp[0][0] = 0  # Base case: 0 elements, 0 subarrays, cost is 0
        
        # Fill the DP table
        for i in range(1, n + 1):  # i is 1-based index up to n
            # The maximum number of subarrays m cannot exceed i
            max_m = i
            for m in range(1, max_m + 1):
                for j in range(i):
                    prev_m = m - 1
                    if dp[j][prev_m] == float('inf'):
                        continue
                    current_sum_num = pre_num[i] - pre_num[j]
                    current_sum_cost = pre_cost[i] - pre_cost[j]
                    total = dp[j][prev_m] + (current_sum_num + k * m) * current_sum_cost
                    if total < dp[i][m]:
                        dp[i][m] = total
        
        # The result is the minimum value among all possible number of subarrays m >= 1
        return min(dp[n][m] for m in range(1, n + 1))