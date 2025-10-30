class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        min_costs = [float('inf')] * n
        total_cost = 0
        for i in range(n):
            min_costs[i] = nums[i]
        
        for k in range(1, n):
            cost_for_k_ops = k * x
            for i in range(n):
                new_index = (i + k) % n
                min_costs[i] = min(min_costs[i], nums[new_index])
        
        for cost in min_costs:
            total_cost += cost
        
        return total_cost