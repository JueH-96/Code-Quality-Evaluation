from typing import List

class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        min_costs = [float('inf')] * n
        total_cost = sum(nums)
        
        for i in range(n):
            min_costs[i] = min(nums[i], min_costs[i-1] if i > 0 else float('inf'))
            total_cost = min(total_cost, min_costs[i] * n + i * x)
        
        return total_cost