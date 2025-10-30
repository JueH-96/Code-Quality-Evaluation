from typing import List

class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i] * (-1 if i % 2 else 1)

        max_cost = prefix_sum[-1]
        for i in range(n - 1):
            for j in range(i + 1, n):
                total_cost = prefix_sum[i + 1] + (prefix_sum[j + 1] - prefix_sum[i + 1]) + (prefix_sum[-1] - prefix_sum[j + 1])
                max_cost = max(max_cost, total_cost)

        return max_cost