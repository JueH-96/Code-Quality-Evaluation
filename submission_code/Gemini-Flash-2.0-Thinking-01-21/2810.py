from typing import List

class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        min_costs = float('inf')
        for operations_count in range(n):
            current_cost = operations_count * x
            min_prices = [float('inf')] * n
            for i in range(n):
                current_type = (i + operations_count) % n
                min_prices[current_type] = min(min_prices[current_type], nums[i])
            chocolate_cost = sum(min_prices)
            total_cost = current_cost + chocolate_cost
            min_costs = min(min_costs, total_cost)
        
        # Consider 0 operations case separately
        cost_no_ops = sum(nums)
        min_costs = min(min_costs, cost_no_ops)
        
        return min_costs