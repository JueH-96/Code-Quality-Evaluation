class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        min_cost = float('inf')
        
        # Try to collect chocolates starting from each type
        for start in range(n):
            current_cost = 0
            for i in range(n):
                # Calculate the cost of collecting the chocolate of type (start + i) % n
                current_cost += nums[(start + i) % n]
                # Add the cost of operations needed to reach this type
                if i > 0:
                    current_cost += x
            # Update the minimum cost
            min_cost = min(min_cost, current_cost)
        
        return min_cost