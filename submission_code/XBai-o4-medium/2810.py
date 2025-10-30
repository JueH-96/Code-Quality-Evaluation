class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        min_costs = list(nums)
        result = sum(min_costs)  # Cost for K=0 operations
        
        for K in range(1, n):
            for i in range(n):
                pos = (i + K) % n
                if nums[pos] < min_costs[i]:
                    min_costs[i] = nums[pos]
            current_sum = sum(min_costs)
            current_cost = current_sum + x * K
            if current_cost < result:
                result = current_cost
        
        return result