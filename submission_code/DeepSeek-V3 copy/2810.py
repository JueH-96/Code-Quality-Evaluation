class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        # Initialize the minimum cost for each type
        min_cost = [float('inf')] * n
        total_min_cost = float('inf')
        
        # Iterate over the number of possible rotations (0 to n-1)
        for rotate in range(n):
            current_cost = 0
            # Calculate the cost for each type after 'rotate' operations
            for i in range(n):
                # The type after 'rotate' operations is (i + rotate) % n
                type_after_rotate = (i + rotate) % n
                # Update the minimum cost for this type
                min_cost[i] = min(min_cost[i], nums[type_after_rotate])
            # Calculate the total cost: sum of min_cost plus rotate * x
            total_cost = sum(min_cost) + rotate * x
            # Update the overall minimum cost
            total_min_cost = min(total_min_cost, total_cost)
        
        return total_min_cost