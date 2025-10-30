from typing import List

class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        min_cost = float('inf')
        
        # Initialize the minimum cost array
        min_costs = nums[:]
        
        for k in range(n):
            current_cost = k * x
            for i in range(n):
                min_costs[i] = min(min_costs[i], nums[(i + k) % n])
                current_cost += min_costs[i]
            
            min_cost = min(min_cost, current_cost)
        
        return min_cost