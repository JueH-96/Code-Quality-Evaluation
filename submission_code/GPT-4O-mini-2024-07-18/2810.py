class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        total_cost = sum(nums)
        
        # Initialize the minimum cost with the total cost without any operations
        min_cost = total_cost
        
        # Calculate the cost for each possible number of operations
        for k in range(1, n):
            # Cost of performing k operations
            operation_cost = k * x
            
            # Calculate the cost of collecting chocolates after k operations
            current_cost = operation_cost
            for i in range(n):
                current_cost += nums[(i + k) % n]
            
            # Update the minimum cost if the current cost is lower
            min_cost = min(min_cost, current_cost)
        
        return min_cost