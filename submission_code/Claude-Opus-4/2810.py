class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        # min_cost[i] represents the minimum cost to collect type i across all operations
        min_cost = nums[:]  # Initially, cost to collect type i is nums[i]
        
        total_min = float('inf')
        
        # Try k operations where k = 0 to n-1
        for k in range(n):
            # After k operations, update minimum costs
            if k > 0:
                for type_needed in range(n):
                    # After k operations, type_needed can be collected from index (type_needed - k) % n
                    index = (type_needed - k) % n
                    min_cost[type_needed] = min(min_cost[type_needed], nums[index])
            
            # Calculate total cost with k operations
            total = k * x + sum(min_cost)
            total_min = min(total_min, total)
        
        return total_min