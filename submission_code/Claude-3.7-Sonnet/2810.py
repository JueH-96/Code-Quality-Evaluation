class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        
        # Initialize with best costs when no operations are performed
        best_costs = nums.copy()
        
        # Initialize result with cost of collecting all types without operations
        result = sum(best_costs)
        
        # Try all possible numbers of operations from 1 to n-1
        for k in range(1, n):
            operation_cost = k * x
            
            # Update the minimum cost to collect each type after k operations
            for t in range(n):
                # Position of type t after k operations
                pos = (t - k + n) % n
                best_costs[t] = min(best_costs[t], nums[pos])
            
            # Update the result with current collection cost + operation cost
            collection_cost = sum(best_costs)
            result = min(result, collection_cost + operation_cost)
        
        return result