class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        min_cost = sum(nums)  # Start with the cost of buying all chocolates without any operations.
        current_cost = min_cost
        
        # Calculate the cost if we start buying from the i-th chocolate.
        for i in range(1, n):
            # Update the current cost by adding the operation cost and subtracting the cost of the chocolate
            # that is now out of the sequence after the operation.
            current_cost += x - nums[n - i]
            # Update the minimum cost if the current cost is lower.
            min_cost = min(min_cost, current_cost)
        
        return min_cost