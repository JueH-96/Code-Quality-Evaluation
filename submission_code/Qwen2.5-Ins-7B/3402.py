from typing import List

class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        mod = 10**9 + 7
        nums.sort()
        n = len(nums)
        prefix_sum = [0]
        for num in nums:
            prefix_sum.append(prefix_sum[-1] + num)
        
        def cost_to_equalize(target: int) -> int:
            total_cost = 0
            for i, num in enumerate(nums):
                if num < target:
                    total_cost += (target - num) * (prefix_sum[i + 1] - prefix_sum[i]) * cost1
                elif num > target:
                    total_cost += (num - target) * (prefix_sum[-1] - prefix_sum[i]) * cost1
                    total_cost -= (num - target) * (num - target - 1) // 2 * cost2
            return total_cost % mod
        
        min_cost = float('inf')
        for i in range(n):
            min_cost = min(min_cost, cost_to_equalize(nums[i]))
        
        return min_cost