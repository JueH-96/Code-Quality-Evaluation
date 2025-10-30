class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        min_cost = float('inf')
        
        # Try performing 0 to n-1 operations
        for k in range(n):
            cost = k * x  # Cost of operations
            
            # After k operations, to collect type j, we look at position (j - k) mod n
            for j in range(n):
                pos = (j - k) % n
                cost += nums[pos]
            
            min_cost = min(min_cost, cost)
        
        return min_cost