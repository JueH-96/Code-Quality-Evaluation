class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        min_total_cost = float('inf')
        
        # min_cost_to_collect[t] stores the minimum cost to collect type t 
        # with at most K operations performed so far
        min_cost_to_collect = nums[:]  # Initially with 0 operations
        
        for K in range(n):
            # Calculate total cost with exactly K operations
            total_cost = K * x + sum(min_cost_to_collect)
            min_total_cost = min(min_total_cost, total_cost)
            
            # Update for next iteration: consider K+1 operations
            if K < n - 1:
                for t in range(n):
                    # After K+1 operations, type t will be at position (t-(K+1))%n
                    pos = (t - (K + 1)) % n
                    min_cost_to_collect[t] = min(min_cost_to_collect[t], nums[pos])
        
        return min_total_cost