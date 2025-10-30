class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        # min_cost[i] tracks the minimum cost to collect chocolate type i
        # considering all operations done so far
        min_cost = nums[:]
        
        # Start with the cost of collecting all chocolates without any operations
        result = sum(min_cost)
        
        # Try performing 1, 2, ..., n-1 operations
        for k in range(1, n):
            # After k operations, chocolate type i can be found at index (i - k) % n
            for i in range(n):
                min_cost[i] = min(min_cost[i], nums[(i - k) % n])
            
            # Calculate total cost: k operations + sum of minimum costs
            result = min(result, k * x + sum(min_cost))
        
        return result