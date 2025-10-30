from typing import List

class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        min_costs = nums.copy()
        for i in range(1, n):
            for j in range(n):
                min_costs[j] = min(min_costs[j], nums[(j + i) % n])
        
        result = float('inf')
        for i in range(n):
            total_cost = sum(min_costs) - min_costs[i] + nums[i]
            total_cost += i * x
            result = min(result, total_cost)
        
        return result